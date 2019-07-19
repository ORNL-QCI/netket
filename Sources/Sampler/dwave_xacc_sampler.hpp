// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_DWAVE_XACC_SAMPLER_HPP
#define NETKET_DWAVE_XACC_SAMPLER_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

#include "Embedding.hpp"
#include "EmbeddingAlgorithm.hpp"
#include "IRGenerator.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

namespace netket {

// Metropolis sampling generating local moves in hilbert space
class DWaveSampler : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;

  int nstates_;
  std::vector<double> localstates_;

  int sweep_size_;

 protected:
  std::shared_ptr<xacc::Function> amplitudeRbm;
  std::shared_ptr<xacc::Function> phaseRbm;
  std::shared_ptr<xacc::Accelerator> dwave;
  xacc::quantum::Embedding embedding;
  int nh = 0;

 public:
  explicit DWaveSampler(AbstractMachine& psi)
      : AbstractSampler(psi), nv_(GetHilbert().Size()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);

    // Get the number of hidden units
    nh = dynamic_cast<RbmSpinPhase&>(GetMachine()).Nhidden();

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Dwave sampler works only for discrete "
          "Hilbert spaces");
    }

    accept_.resize(1);
    moves_.resize(1);

    nstates_ = GetHilbert().LocalSize();
    localstates_ = GetHilbert().LocalStates();

    Reset(true);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    // Get the D-Wave QPU
    dwave = xacc::getAccelerator("dwave");

    // Create the RBM IR Generator
    auto rbmGenerator = xacc::getService<xacc::IRGenerator>("rbm");

    // Create RBMs for Amplitude and Phase
    amplitudeRbm =
        rbmGenerator->generate({{"visible-units", nv_}, {"hidden-units", nh}});
    phaseRbm =
        rbmGenerator->generate({{"visible-units", nv_}, {"hidden-units", nh}});

    /// ---
    // Create the Hardware graph for minor graph embedding
    auto hardwareconnections = dwave->getAcceleratorConnectivity();
    std::set<int> nUniqueBits;
    for (auto& edge : hardwareconnections) {
      nUniqueBits.insert(edge.first);
      nUniqueBits.insert(edge.second);
    }

    int nBits = *std::max_element(nUniqueBits.begin(), nUniqueBits.end()) + 1;

    auto hardware = xacc::getService<xacc::Graph>("boost-ugraph");
    for (int i = 0; i < nBits; i++) {
      std::map<std::string, xacc::InstructionParameter> m{{"bias", 1.0}};
      hardware->addVertex(m);
    }

    for (auto& edge : hardwareconnections) {
      hardware->addEdge(edge.first, edge.second);
    }
    //----

    // Get just one of the RBM graphs
    // std::cout << "HELLO:\n";
    // auto amplitudeGraph = amplitudeRbm->toGraph();
    // std::cout << "HOWDY\n";

    int maxBitIdx = 0;
    for (auto inst : amplitudeRbm->getInstructions()) {
      if (inst->name() == "dw-qmi") {
        auto qbit1 = inst->bits()[0];
        auto qbit2 = inst->bits()[1];
        if (qbit1 > maxBitIdx) {
          maxBitIdx = qbit1;
        }
        if (qbit2 > maxBitIdx) {
          maxBitIdx = qbit2;
        }
      }
    }

    auto problemGraph = xacc::getService<xacc::Graph>("boost-ugraph");
    for (int i = 0; i < maxBitIdx + 1; i++) {
      std::map<std::string, xacc::InstructionParameter> m{{"bias", 1.0}};
      problemGraph->addVertex(m);
    }

    for (auto inst : amplitudeRbm->getInstructions()) {
      if (inst->name() == "dw-qmi") {
        auto qbit1 = inst->bits()[0];
        auto qbit2 = inst->bits()[1];
        if (qbit1 != qbit2) {
          problemGraph->addEdge(qbit1, qbit2, 1.0);
        }
      }
    }

    // Compute the minor graph embedding
    auto a = xacc::getService<xacc::quantum::EmbeddingAlgorithm>("cmr");
    embedding = a->embed(problemGraph, hardware);

    // Create 2 D-Wave Function that
    InfoMessage() << "DWave sampler is ready " << std::endl;
  }

  void Reset(bool initrandom) override {
    if (initrandom) {
      GetHilbert().RandomVals(v_, this->GetRandomEngine());
    }

    GetMachine().InitLookup(v_, lt_);

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    auto qbits1 = xacc::qalloc(2000);
    auto qbits2 = xacc::qalloc(2000);

    qbits1->addExtraInfo("embedding", embedding);
    qbits2->addExtraInfo("embedding", embedding);

    auto params = GetMachine().GetParameters();
    auto half = params.size() / 2;
    Eigen::VectorXcd rbm1(half), rbm2(half);

    // this is [visible | hidden | weights || visible | hidden | weights]
    rbm1.segment(0, nv_) = params.segment(0, nv_);
    rbm2.segment(0, nv_) = params.segment(half, nv_);
    rbm1.segment(nv_, nh) = params.segment(nv_, nh);
    rbm2.segment(nv_, nh) = params.segment(half + nv_, nh);

    const int initw = nv_ + nh;

    rbm1.segment(initw, nv_ * nh) = params.segment(initw, nv_ * nh);
    rbm2.segment(initw, nv_ * nh) = params.segment(half + initw, nv_ * nh);

    auto rbm1_tmp = rbm1.real();
    auto rbm2_tmp = rbm2.real();
    std::vector<double> rbm1vec(rbm1_tmp.data(),
                                rbm1_tmp.data() + rbm1_tmp.size());
    std::vector<double> rbm2vec(rbm2_tmp.data(),
                                rbm2_tmp.data() + rbm2_tmp.size());

    auto amp_tmp = amplitudeRbm->operator()(rbm1vec);
    auto phase_tmp = phaseRbm->operator()(rbm1vec);

    dwave->execute(qbits1, amp_tmp);
    dwave->execute(qbits2, phase_tmp);

    // Task for Sindhu
    // * Unembed the counts...
    // * Map counts to what they need for sampling
    qbits1->print();
    exit(0);

    // My thought is to re-use the Metropolis Local stuff
    // and use all the bit strings from D-Wave
    std::vector<int> tochange(1);
    std::vector<double> newconf(1);

    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distrs(0, nv_ - 1);
    std::uniform_int_distribution<int> diststate(0, nstates_ - 1);

    for (int i = 0; i < sweep_size_; i++) {
      // picking a random site to be changed
      int si = distrs(this->GetRandomEngine());
      assert(si < nv_);
      tochange[0] = si;

      // picking a random state
      int newstate = diststate(this->GetRandomEngine());
      newconf[0] = localstates_[newstate];

      // make sure that the new state is not equal to the current one
      while (std::abs(newconf[0] - v_(si)) <
             std::numeric_limits<double>::epsilon()) {
        newstate = diststate(this->GetRandomEngine());
        newconf[0] = localstates_[newstate];
      }

      const auto lvd = GetMachine().LogValDiff(v_, tochange, newconf, lt_);
      double ratio = this->GetMachineFunc()(std::exp(lvd));

#ifndef NDEBUG
      const auto psival1 = GetMachine().LogVal(v_);
      if (std::abs(
              std::exp(GetMachine().LogVal(v_) - GetMachine().LogVal(v_, lt_)) -
              1.) > 1.0e-8) {
        std::cerr << GetMachine().LogVal(v_) << "  and LogVal with Lt is "
                  << GetMachine().LogVal(v_, lt_) << std::endl;
        std::abort();
      }
#endif

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        accept_[0] += 1;
        GetMachine().UpdateLookup(v_, tochange, newconf, lt_);
        GetHilbert().UpdateConf(v_, tochange, newconf);

#ifndef NDEBUG
        const auto psival2 = GetMachine().LogVal(v_);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << GetMachine().LogVal(v_, lt_) << std::endl;
          std::abort();
        }
#endif
      }
      moves_[0] += 1;
    }
  }

  const Eigen::VectorXd& Visible() const noexcept override { return v_; }

  void SetVisible(const Eigen::VectorXd& v) override { v_ = v; }

  AbstractMachine::VectorType DerLogVisible() override {
    return GetMachine().DerLog(v_, lt_);
  }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};

}  // namespace netket

#endif

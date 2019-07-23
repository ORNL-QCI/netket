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

#ifndef NETKET_DWAVE_SAMPLER_HPP
#define NETKET_DWAVE_SAMPLER_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

#include "EmbeddingAlgorithm.hpp"
#include "IRGenerator.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

namespace netket {

// Exact sampling using heat bath, mostly for testing purposes on small systems
class DWaveSampler : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  const HilbertIndex& hilbert_index_;

  const int dim_;

  std::discrete_distribution<int> dist_;

  std::vector<Complex> logpsivals_;
  std::vector<double> psivals_;

 protected:
  std::shared_ptr<xacc::Function> _rbm;
  std::shared_ptr<xacc::Accelerator> dwave;
  xacc::quantum::Embedding embedding;
  int nh = 0;

 public:
  explicit DWaveSampler(AbstractMachine& psi)
      : AbstractSampler(psi),
        nv_(GetHilbert().Size()),
        hilbert_index_(GetHilbert().GetIndex()),
        dim_(hilbert_index_.NStates()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);
    try {
      nh = dynamic_cast<RbmSpinPhase&>(GetMachine()).Nhidden();
    } catch (const std::bad_cast& e) {
      // Cast failed, try real
      nh = dynamic_cast<RbmSpin&>(GetMachine()).Nhidden();
    }

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "DWave sampler works only for discrete "
          "Hilbert spaces");
    }

    accept_.resize(1);
    moves_.resize(1);

    // Get the D-Wave QPU
    dwave = xacc::getAccelerator("dwave");

    // Create the RBM IR Generator
    auto rbmGenerator = xacc::getService<xacc::IRGenerator>("rbm");

    // Create RBMs for Amplitude and Phase
    _rbm =
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

    int maxBitIdx = 0;
    for (auto inst : _rbm->getInstructions()) {
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

    for (auto inst : _rbm->getInstructions()) {
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

    // Reset(true);

    InfoMessage() << "DWave sampler is ready " << std::endl;
  }

  void Reset(bool initrandom) override {
    if (initrandom) {
      GetHilbert().RandomVals(v_, this->GetRandomEngine());
    }

    psivals_.resize(dim_);
    //   auto v = hilbert_index_.NumberToState(4);
    //   std::cout << nh << ", HELLO: \n" << v << "\n";

    // auto vv = (Eigen::VectorXd::Ones(4) + v ) / 2;
    // std::cout << "HOWDY:\n" << vv << "\n";
    // exit(0);
    auto qbits1 = xacc::qalloc(2000);

    qbits1->addExtraInfo("embedding", embedding);

    auto params = GetMachine().GetParameters();
    auto half = params.size() / 2;
    Eigen::VectorXcd rbm1(half);

    // this is [visible | hidden | weights || visible | hidden | weights]
    rbm1.segment(0, nv_) = params.segment(0, nv_);
    rbm1.segment(nv_, nh) = params.segment(nv_, nh);

    const int initw = nv_ + nh;

    rbm1.segment(initw, nv_ * nh) = params.segment(initw, nv_ * nh);

    auto rbm1_tmp = rbm1.real();
    std::vector<double> rbm1vec(rbm1_tmp.data(),
                                rbm1_tmp.data() + rbm1_tmp.size());

    auto amp_tmp = _rbm->operator()(rbm1vec);

    dwave->execute(qbits1, amp_tmp);

    // Task for Sindhu
    // * Unembed the counts...
    // * Map counts to what they need for sampling
    // qbits1->print();
    auto results = qbits1->getMeasurementCounts();
    std::map<std::string, double> visibleCounts;
    double totalCounts = 0;
    for (auto& kv : results) {
        totalCounts += kv.second;
        auto vis = kv.first.substr(0, nv_);
        if (visibleCounts.count(vis)) {
            visibleCounts[vis] += kv.second;
        } else {
            visibleCounts.insert({vis, kv.second});
        }
    }
    // exit(0);

    // for (auto & kv : visibleCounts) {
    //     std::cout << "VIS: " << kv.first << ", " << kv.second << "\n";
    // }
    // FIXME psivals_ from counts distribution...
    for (int i = 0; i < dim_; ++i) {
      auto v = hilbert_index_.NumberToState(i);
      auto vv = (Eigen::VectorXi::Ones(nv_) + v.cast<int>() ) / 2;

      std::string bitStr = "";
      for (int j = 0; j < nv_; j++) bitStr += std::to_string(vv(j));

      if (visibleCounts.count(bitStr)) {
        psivals_[i] = visibleCounts[bitStr] / totalCounts;
      } else {
        psivals_[i] = 0.0;
      }
    }

    dist_ = std::discrete_distribution<int>(psivals_.begin(), psivals_.end());

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    int newstate = dist_(this->GetRandomEngine());
    v_ = hilbert_index_.NumberToState(newstate);

    accept_(0) += 1;
    moves_(0) += 1;
  }

  const Eigen::VectorXd& Visible() const noexcept override { return v_; }

  void SetVisible(const Eigen::VectorXd& v) override { v_ = v; }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }

  void SetMachineFunc(MachineFunction machine_func) override {
    AbstractSampler::SetMachineFunc(machine_func);
    Reset(true);
  }
};

}  // namespace netket

#endif

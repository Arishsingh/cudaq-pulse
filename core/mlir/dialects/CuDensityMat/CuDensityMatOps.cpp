// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

// Declarations only — .cpp.inc definitions live in CuDensityMatDialect.cpp
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

using namespace mlir;

namespace cudm {

LogicalResult CreateStateOp::verify() {
  auto extents = getModeExtents();
  if (extents.empty())
    return emitOpError("mode_extents must have at least one element");
  for (int64_t e : extents) {
    if (e < 2)
      return emitOpError("each mode extent must be >= 2, got ") << e;
  }
  return success();
}

LogicalResult CreateElementaryOpOp::verify() {
  auto extents = getModeExtents();
  if (extents.empty())
    return emitOpError("mode_extents must have at least one element");
  for (int64_t e : extents) {
    if (e < 2)
      return emitOpError("each mode extent must be >= 2, got ") << e;
  }
  auto offsets = getDiagonalOffsets();
  if (getSparsity() != Sparsity::Multidiagonal) {
    if (offsets)
      return emitOpError("diagonal_offsets are for a multi-diagonal operator");
    return success();
  }
  if (extents.size() != 1)
    return emitOpError("a multi-diagonal operator acts on one mode");
  if (!offsets || offsets->empty())
    return emitOpError("a multi-diagonal operator needs diagonal_offsets");
  for (auto [i, offset] : llvm::enumerate(*offsets)) {
    if (offset <= -extents[0] || offset >= extents[0])
      return emitOpError("diagonal offset ")
             << offset << " is outside a mode of " << extents[0] << " levels";
    if (i > 0 && offset <= (*offsets)[i - 1])
      return emitOpError("diagonal_offsets must ascend");
  }
  DenseElementsAttr values;
  if (matchPattern(getTensorData(), m_Constant(&values)) &&
      values.getNumElements() !=
          2 * static_cast<int64_t>(offsets->size()) * extents[0])
    return emitOpError("multi-diagonal data must hold ")
           << offsets->size() << " complex vectors of " << extents[0]
           << " elements";
  return success();
}

LogicalResult AppendElementaryProductOp::verify() {
  if (getElemOps().empty())
    return emitOpError("at least one elementary operator must be provided");
  auto modes = getModesActedOn();
  auto duality = getDuality();
  // An elementary operator acts on as many modes as it has extents; one made
  // elsewhere acts on at least one.
  size_t operatorModes = 0;
  bool allKnown = true;
  for (Value elementary : getElemOps()) {
    if (auto create = elementary.getDefiningOp<CreateElementaryOpOp>())
      operatorModes += create.getModeExtents().size();
    else {
      operatorModes += 1;
      allKnown = false;
    }
  }
  if (allKnown ? modes.size() != operatorModes : modes.size() < operatorModes)
    return emitOpError("modes_acted_on size (")
           << modes.size() << ") must match the modes of its elementary "
           << "operators (" << operatorModes << ")";
  if (duality.size() != modes.size())
    return emitOpError("duality size (")
           << duality.size() << ") must match modes_acted_on size ("
           << modes.size() << ")";
  return success();
}

LogicalResult OperatorAppendTermOp::verify() {
  int32_t d = getDuality();
  if (d != 0 && d != 1)
    return emitOpError("duality must be 0 (as built) or 1 (every mode "
                       "action flipped between ket and bra), got ")
           << d;
  return success();
}

LogicalResult OperatorPrepareActionOp::verify() {
  const int64_t limit = getWorkspaceLimitAttr().getInt();
  if (limit < 0)
    return emitOpError("workspace_limit must be non-negative, got ") << limit;
  return success();
}

LogicalResult OperatorComputeActionOp::verify() { return success(); }

LogicalResult ExpectationComputeOp::verify() { return success(); }

LogicalResult SSEEvolveOp::verify() {
  const int64_t trajectories = getTrajectoryCountAttr().getInt();
  if (trajectories < 1)
    return emitOpError("trajectory_count must be >= 1, got ") << trajectories;
  const int64_t steps = getStepCountAttr().getInt();
  if (steps < 1)
    return emitOpError("step_count must be >= 1, got ") << steps;
  double tStart = getTStart().convertToDouble();
  double tEnd = getTEnd().convertToDouble();
  if (tEnd <= tStart)
    return emitOpError("t_end (")
           << tEnd << ") must be greater than t_start (" << tStart << ")";
  return success();
}

} // namespace cudm

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.cpp.inc"

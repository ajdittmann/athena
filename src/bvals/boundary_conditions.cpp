//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in
 * the code distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh.hpp"
#include "../fluid.hpp"
#include "prototypes.hpp"
#include "boundary_conditions.hpp"

//======================================================================================
/*! \file boundary_conditions.cpp
 *  \brief boundary conditions for fluid (quantities in ghost zones) on each edge
 *====================================================================================*/

// constructor -- set BC function pointers based on integer flags read from input file

BoundaryConditions::BoundaryConditions(ParameterInput *pin, Fluid *pf)
{
  std::stringstream msg;
  int flag;
  pmy_fluid = pf;

// Set BC function pointers for each of the 6 boundaries in turn -----------------------
// Inner x1

  flag = pin->GetOrAddInteger("mesh","ix1_bc",0);
  switch(flag){
    case 1:
      FluidInnerX1_ = ReflectInnerX1;
    break;
    default:
      msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
          << "Boundary condition flag ix1_bc=" << flag << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    break;
   }

// Outer x1

  flag = pin->GetOrAddInteger("mesh","ox1_bc",0);
  switch(flag){
    case 1:
      FluidOuterX1_ = ReflectOuterX1;
    break;
    default:
      msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
          << "Boundary condition flag ox1_bc=" << flag << " not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    break;
  }

// Inner x2

  int nx2  = pin->GetInteger("mesh","nx2");
  if (nx2 > 1) {
    flag = pin->GetOrAddInteger("mesh","ix2_bc",0);
    switch(flag){
      case 1:
        FluidInnerX2_ = ReflectInnerX2;
      break;
      default:
        msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
            << "Boundary condition flag ix2_bc=" << flag << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      break;
     }

// Outer x2

    flag = pin->GetOrAddInteger("mesh","ox2_bc",0);
    switch(flag){
      case 1:
        FluidOuterX2_ = ReflectOuterX2;
      break;
      default:
        msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
            << "Boundary condition flag ox2_bc=" << flag << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      break;
    }
  }

// Inner x3

  int nx3  = pin->GetInteger("mesh","nx3");
  if (nx3 > 1) {
    flag = pin->GetOrAddInteger("mesh","ix3_bc",0);
    switch(flag){
      case 1:
        FluidInnerX3_ = ReflectInnerX3;
      break;
      default:
        msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
            << "Boundary condition flag ix3_bc=" << flag << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      break;
     }

// Outer x3

    flag = pin->GetOrAddInteger("mesh","ox3_bc",0);
    switch(flag){
      case 1:
        FluidOuterX3_ = ReflectOuterX3;
      break;
      default:
        msg << "### FATAL ERROR in BoundaryConditions constructor" << std::endl
            << "Boundary condition flag ox3_bc=" << flag << " not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      break;
    }
  }

}

// destructor

BoundaryConditions::~BoundaryConditions()
{
}

//--------------------------------------------------------------------------------------
/*! \fn void SetBoundaryValues()
 *  \brief Calls BC functions using appropriate function pointers to set ghost zones.  
 */

void BoundaryConditions::SetBoundaryValues(AthenaArray<Real> &a)
{

// Boundary Conditions in x1-direction

  (*(FluidInnerX1_))(pmy_fluid,a);
  (*(FluidOuterX1_))(pmy_fluid,a);

// Boundary Conditions in x2-direction 

  if (pmy_fluid->pmy_block->block_size.nx2 > 1){

    (*(FluidInnerX2_))(pmy_fluid,a);
    (*(FluidOuterX2_))(pmy_fluid,a);

  }

// Boundary Conditions in x3-direction 

  if (pmy_fluid->pmy_block->block_size.nx3 > 1){

    (*(FluidInnerX3_))(pmy_fluid,a);
    (*(FluidOuterX3_))(pmy_fluid,a);

  }

  return;
}
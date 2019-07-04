/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#ifndef _aspect_mesh_deformation_diffusion_h
#define _aspect_mesh_deformation_diffusion_h

#include <aspect/mesh_deformation/interface.h>

#include <aspect/simulator_access.h>
#include <aspect/simulator/assemblers/interface.h>

#include <aspect/geometry_model/initial_topography_model/interface.h>


namespace aspect
{
  using namespace dealii;


  namespace MeshDeformation
  {
    /**
     * A plugin that computes the deformation of surface
     * vertices according to the solution of the flow problem.
     * In particular this means if the surface of the domain is
     * left open to flow, this flow will carry the mesh with it.
     */
    template<int dim>
    class Diffusion : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Diffusion();

        /**
         * Initialize function, which connects the set_assemblers function
         * to the appropriate Simulator signal.
         */
        virtual void initialize();

        /**
         * The update function sets the current time.
         */
        virtual void update();

        /**
         * Called by Simulator::set_assemblers() to allow the Diffusion plugin
         * to register its assembler.
         */
        void set_assemblers(const SimulatorAccess<dim> &simulator_access,
                            aspect::Assemblers::Manager<dim> &assemblers) const;

        /**
         * A function that creates constraints for the velocity of certain mesh
         * vertices (e.g. the surface vertices) for a specific boundary.
         * The calling class will respect
         * these constraints when computing the new vertex positions.
         */
        virtual
        void
        compute_velocity_constraints_on_boundary(const DoFHandler<dim> &mesh_deformation_dof_handler,
                                                 ConstraintMatrix &mesh_velocity_constraints,
                                                 std::set<types::boundary_id> boundary_id) const;

        /**
         * Declare parameters for the free surface handling.
         */
        static
        void declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters for the free surface handling.
         */
        void parse_parameters (ParameterHandler &prm);

      private:
        /**
         * Compute the surface velocity from a difference
         * in surface height given by the solution of 
         * the hillslope diffusion problem. 
         */
        void diffuse_boundary (const DoFHandler<dim> &free_surface_dof_handler,
                               const IndexSet &mesh_locally_owned,
                               const IndexSet &mesh_locally_relevant,
                               LinearAlgebra::Vector &output,
                               const std::set<types::boundary_id> boundary_id) const;

        /**
         * The hillslope transport coefficient or diffusivity [m2/yr]
         * used in the hillslope diffusion of the deformed
         * surface. TODO Reasonable values lie between X and X.
         */
        double diffusivity;

        /**
         * TODO
         * The diffusion timestep used in case the advection timestep
         * is larger than this timestep.
         */
        double diffusion_time_step;

        /**
         * TODO
         * The amount of model time between applying diffusion
         * of the free surface.
         */
        double time_between_diffusion;

        /**
         * TODO
         * Not used.
         */
        double current_time;
    };
  }
}


#endif

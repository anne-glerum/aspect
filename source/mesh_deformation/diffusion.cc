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


#include <aspect/mesh_deformation/diffusion.h>
#include <aspect/simulator_signals.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/melt.h>
#include <aspect/simulator.h>
#include <aspect/geometry_model/initial_topography_model/zero_topography.h>
#include <aspect/geometry_model/box.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/numerics/vector_tools.h>

namespace aspect
{

  namespace MeshDeformation
  {
    template <int dim>
    Diffusion<dim>::Diffusion()
      :
      diffusivity(0),
      diffusion_time_step(0),
      time_between_diffusion(0),
      current_time(0)
    {}



    template <int dim>
    void
    Diffusion<dim>::initialize ()
    {
    }



    template <int dim>
    void
    Diffusion<dim>::update ()
    {
      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      if (this->convert_output_to_years())
        current_time = (this->get_time() / year_in_seconds);
      else
        current_time = (this->get_time());
    }



    template <int dim>
    void Diffusion<dim>::diffuse_boundary(const DoFHandler<dim> &mesh_deformation_dof_handler,
                                          const IndexSet &mesh_locally_owned,
                                          const IndexSet &mesh_locally_relevant,
                                          LinearAlgebra::Vector &output,
                                          const std::set<types::boundary_id> boundary_ids) const
    {
      // Set up the system to solve
      LinearAlgebra::SparseMatrix mass_matrix;
      LinearAlgebra::Vector system_rhs, solution;
      system_rhs.reinit(mesh_locally_owned, this->get_mpi_communicator());
      solution.reinit(mesh_locally_owned, this->get_mpi_communicator());

      // Set up constraints
      ConstraintMatrix mass_matrix_constraints(mesh_locally_relevant);
      DoFTools::make_hanging_node_constraints(mesh_deformation_dof_handler, mass_matrix_constraints);

      typedef std::set< std::pair< std::pair<types::boundary_id, types::boundary_id>, unsigned int> > periodic_boundary_pairs;
      periodic_boundary_pairs pbp = this->get_geometry_model().get_periodic_boundary_pairs();
      for (periodic_boundary_pairs::iterator p = pbp.begin(); p != pbp.end(); ++p)
        DoFTools::make_periodicity_constraints(mesh_deformation_dof_handler,
                                               (*p).first.first, (*p).first.second, (*p).second, mass_matrix_constraints);
      // TODO What constraints do we want?
      // For now just assume Neumann BC

      // The list of boundary indicators for which we need to set
      // no_normal_flux_constraints, which means all
      // minus the diffusion mesh deformation boundary indicators.
//      std::set< types::boundary_id > x_no_flux_boundary_indicators = this->get_geometry_model().get_used_boundary_indicators();
//      for (std::set<types::boundary_id>::const_iterator p = x_no_flux_boundary_indicators.begin();
//           p != x_no_flux_boundary_indicators.end(); ++p)
//        if (boundary_ids.find(*p) != boundary_ids.end())
//          {
//            x_no_flux_boundary_indicators.erase(*p);
//          }
//
//      // Make the no flux boundary constraints
//      VectorTools::compute_no_normal_flux_constraints (mesh_deformation_dof_handler,
//                                                       /* first_vector_component= */
//                                                       0,
//                                                       x_no_flux_boundary_indicators,
//                                                       mass_matrix_constraints, this->get_mapping());
      mass_matrix_constraints.close();

      // Sparsity of the matrix
#ifdef ASPECT_USE_PETSC
      LinearAlgebra::DynamicSparsityPattern sp(mesh_locally_relevant);

#else
      TrilinosWrappers::SparsityPattern sp (mesh_locally_owned,
                                            mesh_locally_owned,
                                            mesh_locally_relevant,
                                            this->get_mpi_communicator());
#endif
      DoFTools::make_sparsity_pattern (mesh_deformation_dof_handler, sp, mass_matrix_constraints, false,
                                       Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));

#ifdef ASPECT_USE_PETSC
      SparsityTools::distribute_sparsity_pattern(sp,
                                                 mesh_deformation_dof_handler.n_locally_owned_dofs_per_processor(),
                                                 this->get_mpi_communicator(), mesh_locally_relevant);

      sp.compress();
      mass_matrix.reinit (mesh_locally_owned, mesh_locally_owned, sp, this->get_mpi_communicator());
#else
      sp.compress();
      mass_matrix.reinit (sp);
#endif

      // Initialize Gauss-Legendre quadrature for degree+1 quadrature points of the surface faces
      QGauss<dim-1> face_quadrature(mesh_deformation_dof_handler.get_fe().degree+1);
      // Update shape function values and gradients, the quadrature points and the Jacobian x quadrature weights.
      UpdateFlags update_flags = UpdateFlags(update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
      // We want to extract the displacement at the free surface faces of the mesh deformation element.
      FEFaceValues<dim> fs_fe_face_values (this->get_mapping(), mesh_deformation_dof_handler.get_fe(), face_quadrature, update_flags);
      // and to solve on the whole mesh deformation mesh
      // The number of quadrature points on a mesh deformation surface face
      const unsigned int n_fs_face_q_points = fs_fe_face_values.n_quadrature_points;

      // What we need to build our system on the mesh deformation element

      // The nr of shape functions per mesh deformation element
      const unsigned int dofs_per_cell = mesh_deformation_dof_handler.get_fe().dofs_per_cell;

      // Map of local to global cell doff indices
      std::vector<types::global_dof_index> cell_dof_indices (dofs_per_cell);

      // The local rhs vector
      Vector<double> cell_vector (dofs_per_cell);
      // The local matrix
      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);

      // Vector for getting the local dim displacement values
      std::vector<Tensor<1, dim> > displacement_values(n_fs_face_q_points);

      // Vector for getting the local dim initial topography values
      std::vector<Tensor<1, dim> > initial_topography_values(n_fs_face_q_points);

      // The global displacements on the MeshDeformation FE
      LinearAlgebra::Vector displacements = this->get_mesh_deformation_handler().get_mesh_displacements();

      // The global initial topography on the MeshDeformation FE
      LinearAlgebra::Vector initial_topography = this->get_mesh_deformation_handler().get_initial_topography();

      // Do nothing at time zero
      if (this->get_timestep_number() == 0)
        return;

      // An extractor for the dim-valued displacement vectors
      // Later on we will compute the gravity-parallel displacement
      FEValuesExtractors::Vector extract_vertical_displacements(0);

      // An extractor for the dim-valued initial topography vectors
      // Later on we will compute the gravity-parallel displacement
      FEValuesExtractors::Vector extract_initial_topography(0);

      // Cell iterator over the MeshDeformation FE
      typename DoFHandler<dim>::active_cell_iterator
      fscell = mesh_deformation_dof_handler.begin_active(),
      fsendc= mesh_deformation_dof_handler.end();

      // Iterate over all cells to find those at the mesh deformation boundary
      for (; fscell!=fsendc; ++fscell)
        if (fscell->at_boundary() && fscell->is_locally_owned())
          for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            if (fscell->face(face_no)->at_boundary())
              {
                // Boundary indicator of current cell face
                const types::boundary_id boundary_indicator
                  = fscell->face(face_no)->boundary_id();

                // Only apply diffusion to the requested boundaries
                if (boundary_ids.find(boundary_indicator) == boundary_ids.end())
                  continue;

                // Get the global numbers of the local DoFs of the mesh deformation cell
                fscell->get_dof_indices (cell_dof_indices);

                // Recompute values, gradients, etc on the faces
                fs_fe_face_values.reinit (fscell, face_no);

                // Extract the displacement values
                fs_fe_face_values[extract_vertical_displacements].get_function_values (displacements, displacement_values);

                // Extract the initial topography values
                fs_fe_face_values[extract_initial_topography].get_function_values (initial_topography, initial_topography_values);

                // Reset local rhs and matrix
                cell_vector = 0;
                cell_matrix = 0;

                // Loop over the quadrature points of the current face
                for (unsigned int point=0; point<n_fs_face_q_points; ++point)
                  {
                    // Get the gravity vector to compute the outward direction of displacement
                    Tensor<1,dim> direction = this->get_gravity_model().gravity_vector(fs_fe_face_values.quadrature_point(point));
                    // Normalize direction vector
                    if (direction.norm() > 0.0)
                      direction *= 1./direction.norm();
                    // TODO this is only correct for box geometries
                    else
                      direction[dim-1] = 1.;

                    // Compute the total displacement in the gravity direction,
                    // i.e. the initial topography + any additional mesh displacement.
                    const double displacement = (displacement_values[point] * direction) + (initial_topography_values[point] * direction);

                    // Loop over the shape functions
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                      {
                        // Assemble the RHS
                        // RHS = M*H_old
                        cell_vector(i) += displacement *
                                          fs_fe_face_values.shape_value (i, point) *
                                          /*fs_fe_face_values.shape_value (j, point) * */
                                          fs_fe_face_values.JxW(point);

                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                          {
                            // Assemble the matrix, for backward first order time discretization:
                            // Matrix := (M+dt*K) = (M+dt*B^T*kappa*B)
                            // To project onto the tangent space of the surface,
                            // we define the projection P:= I- n x n,
                            // with I the unit tensor and n the unit normal to the surface.
                            // The surface gradient then is P times the usual gradient of the shape functions.
                            const Tensor<2, dim, double> projection = unit_symmetric_tensor<dim>() -
                                                                      outer_product(fs_fe_face_values.normal_vector(point), fs_fe_face_values.normal_vector(point));

                            cell_matrix(i,j) +=
                              (
                                this->get_timestep() * diffusivity *
                                projection * fs_fe_face_values.shape_grad(i, point) * projection * fs_fe_face_values.shape_grad(j,point) +
                                fs_fe_face_values.shape_value (i, point) * fs_fe_face_values.shape_value (j, point)
                              )
                              * fs_fe_face_values.JxW(point);
                          }
                      }

                  }

                mass_matrix_constraints.distribute_local_to_global (cell_matrix, cell_vector,
                                                                    cell_dof_indices, mass_matrix, system_rhs, false);
              }

      system_rhs.compress (VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);

      // Jacobi seems to be fine here.  Other preconditioners (ILU, IC) run into trouble
      // because the matrix is mostly empty, since we don't touch internal vertices.
      LinearAlgebra::PreconditionJacobi preconditioner_mass;
      preconditioner_mass.initialize(mass_matrix);

      SolverControl solver_control(5*system_rhs.size(), this->get_parameters().linear_stokes_solver_tolerance*system_rhs.l2_norm());
      SolverCG<LinearAlgebra::Vector> cg(solver_control);
      cg.solve (mass_matrix, solution, system_rhs, preconditioner_mass);

      // Distribute constraints on mass matrix
      mass_matrix_constraints.distribute (solution);

      // The solution contains the new displacements, we need to return a velocity.
      // Therefore, we compute v=d_displacement/d_t.
      // d_displacement are the new mesh node locations
      // minus the old locations.
      LinearAlgebra::Vector d_displacement(mesh_locally_owned, this->get_mpi_communicator());
      d_displacement = solution;
      d_displacement -= displacements;
      d_displacement -= initial_topography;
      // The velocity
      d_displacement /= this->get_timestep();

      output = d_displacement;
    }



    /**
     * A function that creates constraints for the velocity of certain mesh
     * vertices (e.g. the surface vertices) for a specific boundary.
     * The calling class will respect
     * these constraints when computing the new vertex positions.
     */
    template <int dim>
    void
    Diffusion<dim>::compute_velocity_constraints_on_boundary(const DoFHandler<dim> &mesh_deformation_dof_handler,
                                                             ConstraintMatrix &mesh_velocity_constraints,
                                                             std::set<types::boundary_id> boundary_id) const
    {
      LinearAlgebra::Vector boundary_velocity;

      const IndexSet mesh_locally_owned = mesh_deformation_dof_handler.locally_owned_dofs();
      IndexSet mesh_locally_relevant;
      DoFTools::extract_locally_relevant_dofs (mesh_deformation_dof_handler,
                                               mesh_locally_relevant);
      boundary_velocity.reinit(mesh_locally_owned, mesh_locally_relevant,
                               this->get_mpi_communicator());

      diffuse_boundary(mesh_deformation_dof_handler, mesh_locally_owned,
                       mesh_locally_relevant, boundary_velocity, boundary_id);

      // now insert the relevant part of the solution into the mesh constraints
      IndexSet constrained_dofs;
      DoFTools::extract_boundary_dofs(mesh_deformation_dof_handler,
                                      ComponentMask(dim, true),
                                      constrained_dofs,
                                      boundary_id);

      for (unsigned int i = 0; i < constrained_dofs.n_elements();  ++i)
        {
          types::global_dof_index index = constrained_dofs.nth_index_in_set(i);
          if (mesh_velocity_constraints.can_store_line(index))
            if (mesh_velocity_constraints.is_constrained(index)==false)
              {
                mesh_velocity_constraints.add_line(index);
                mesh_velocity_constraints.set_inhomogeneity(index, boundary_velocity[index]);
              }
        }

    }


    template <int dim>
    void Diffusion<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection ("Mesh deformation");
      {
        prm.enter_subsection ("Diffusion");
        {
          prm.declare_entry("Hillslope transport coefficient", "0.5",
                            Patterns::Double(0),
                            "The hillslope transport coefficient used to "
                            "diffuse the free surface, either as a  "
                            "stabilization step or a to mimic erosional "
                            "and depositional processes. ");
          prm.declare_entry("Diffusion timestep", "2000",
                            Patterns::Double(0),
                            "The timestep used in the solving of the diffusion "
                            "equation. ");
          prm.declare_entry("Time between diffusion", "2000",
                            Patterns::Double(0),
                            "The timestep used in the solving of the diffusion "
                            "equation. ");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void Diffusion<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection ("Mesh deformation");
      {
        prm.enter_subsection ("Diffusion");
        {
          diffusivity = prm.get_double("Hillslope transport coefficient");
          diffusion_time_step = prm.get_double("Diffusion timestep");
          time_between_diffusion = prm.get_double("Time between diffusion");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace MeshDeformation
  {
    ASPECT_REGISTER_MESH_DEFORMATION_MODEL(Diffusion,
                                           "diffusion",
                                           "A plugin that computes the deformation of surface "
                                           "vertices according to the solution of the hillslope diffusion problem. "
                                           "TODO add more documentation.")
  }
}

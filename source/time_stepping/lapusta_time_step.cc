/*
  Copyright (C) 2018 - 2020 by the authors of the ASPECT code.

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


#include <aspect/global.h>
#include <aspect/time_stepping/lapusta_time_step.h>

namespace aspect
{
  namespace TimeStepping
  {
    template <int dim>
    double
    LapustaTimeStep<dim>::execute()
    {
      std::cout << "entered Lapusta time step execute" << std::endl;

      const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                               this->get_parameters().stokes_velocity_degree);
      std::cout << "entered Lapusta time step execute - got quadrature formula" << std::endl;

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values);
      std::cout << "entered Lapusta time step execute - got fe-values" << std::endl;

      const unsigned int n_q_points = quadrature_formula.size();
      std::cout << "entered Lapusta time step execute - got nq-points" << std::endl;

      std::vector<Tensor<1,dim> > velocity_values(n_q_points);
      MaterialModel::MaterialModelInputs<dim> in(n_q_points,
                                                 this->introspection().n_compositional_fields);

      // Do I need "out"?
      MaterialModel::MaterialModelOutputs<dim> out(n_q_points,
                                                   this->introspection().n_compositional_fields);
      std::cout << "entered Lapusta time step execute - got in#n#out" << std::endl;
      
      ComponentMask composition_mask = visco_plastic.get_volumetric_composition_mask();
      std::cout << "entered Lapusta time step execute - got composition mask" << std::endl;

      double min_state_weakening_time_step = 1.e30; //TODO: get the actual max number? Now its just arbitrarily big
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
      std::cout << "entered Lapusta time step execute - cell loop" << std::endl;
            fe_values.reinit (cell);
            in.reinit(fe_values,
                      cell,
                      this->introspection(),
                      this->get_solution());
            // do I need to do this? It is copied from conduction_time_step
            this->get_material_model().evaluate(in, out);

            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);

            double max_local_velocity = 0;

            const double delta_x = cell->minimum_vertex_distance();
            for (unsigned int q=0; q<n_q_points; ++q)
              {
      std::cout << "entered Lapusta time step execute - cell loop - nq-points" << std::endl;
                // TODO in Lapusta, this should be plastic velocity. But just taking the max velocity
                // should be the most conservative approach, so it should be ok...
                max_local_velocity = velocity_values[q].norm();

                const double pressure = in.pressure[q]; //TODO: is this correct to get the pressure? out. has no member pressure...
                std::pair<double,double> delta_theta_max_and_critical_slip_distance = friction_options.compute_delta_theta_max(composition_mask, in.composition[0], in.position[q], delta_x, pressure);
                min_state_weakening_time_step = std::min (min_state_weakening_time_step,
                                                          delta_theta_max_and_critical_slip_distance.first
                                                          * delta_theta_max_and_critical_slip_distance.second / max_local_velocity);
              
              
      std::cout << "entered Lapusta time step execute - cell loop - nq-points - end" << std::endl;}

          }


      /*
      const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                               this->get_parameters().stokes_velocity_degree);

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values);

      const unsigned int n_q_points = quadrature_formula.size();

      std::vector<Tensor<1,dim> > velocity_values(n_q_points);
      std::vector<Tensor<1,dim> > fluid_velocity_values(n_q_points);

      double max_local_speed_over_meshsize = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);

            double max_local_velocity = 0;
            for (unsigned int q=0; q<n_q_points; ++q)
              max_local_velocity = std::max (max_local_velocity,
                                             velocity_values[q].norm());

            if (this->get_parameters().include_melt_transport)
              {
                const FEValuesExtractors::Vector ex_u_f = this->introspection().variable("fluid velocity").extractor_vector();
                fe_values[ex_u_f].get_function_values (this->get_solution(), fluid_velocity_values);

                for (unsigned int q=0; q<n_q_points; ++q)
                  max_local_velocity = std::max (max_local_velocity,
                                                 fluid_velocity_values[q].norm());
              }

            max_local_speed_over_meshsize = std::max(max_local_speed_over_meshsize,
                                                     max_local_velocity
                                                     /
                                                     cell->minimum_vertex_distance());

          }

      const double max_global_speed_over_meshsize
        = Utilities::MPI::max (max_local_speed_over_meshsize, this->get_mpi_communicator());
      */
      double min_lapusta_timestep = std::numeric_limits<double>::max();

      /*      if (max_global_speed_over_meshsize != 0.0)
              min_lapusta_timestep = this->get_parameters().CFL_number / (this->get_parameters().temperature_degree * max_global_speed_over_meshsize);
      */

      //min_lapusta_timestep = std::min(state_weakening_time_step,healing_time_step);
      min_lapusta_timestep = min_state_weakening_time_step;
      std::cout << "entered Lapusta time step execute - end" << std::endl;

      AssertThrow (min_lapusta_timestep > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length was: " + std::to_string(min_lapusta_timestep) + ". "
                              "Please check for non-positive material properties."));

      return min_lapusta_timestep;
    }


  }
}

// explicit instantiations
namespace aspect
{
  namespace TimeStepping
  {
    ASPECT_REGISTER_TIME_STEPPING_MODEL(LapustaTimeStep,
                                        "lapusta time step",
                                        "This model computes the state weakening and state healing "
                                        "time step following Lapusta et al. 2000. It is intended for models "
                                        "with rate and state friction. min dt computed as follows: TODO.... ")
  }
}

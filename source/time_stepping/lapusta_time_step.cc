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
      const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                               this->get_parameters().stokes_velocity_degree);

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values);

      const unsigned int n_q_points = quadrature_formula.size();

      std::vector<Tensor<1,dim> > velocity_values(n_q_points);

      double min_state_weakening_time_step = 1e10;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);

            double max_local_velocity = 0;
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                max_local_velocity = std::max (max_local_velocity,
                                               velocity_values[q].norm());

                const unsigned int j = 0;
                const double G_star = G/(1-nu); //G is shear modulus , nu is poisson ration (0.25 in herrendo)
                const double k_param = 2 / numbers::PI * G_star / cell->minimum_vertex_distance();
                const double RSF_param_a = friction_options.calculate_depth_dependent_a_and_b(position,j).first;
                const double RSF_param_b = friction_options.calculate_depth_dependent_a_and_b(position,j).second;
                const double RSF_param_L = friction_options.get_critical_slip_distance(in.position[q], j); // j is from volume fractions, q is n_evaluation points
                const double pressure;
                const double kLaP = (k_param * RSF_param_L) / (RSF_param_a * pressure);
                const double xi = 0.25 * std::pow((kLaP - (RSF_param_b - RSF_param_a) / RSF_param_a),2) - kLaP;

                double delta_theta_max = 0;
                if (xi > 0)
                  delta_theta_max = std::min(((RSF_param_a * pressure)
                                              / (k_param * RSF_param_L - (RSF_param_b - RSF_param_a)
                                                 * pressure)), 0.2);
                else
                  delta_theta_max = std::min((1 - ((RSF_param_b - RSF_param_a) * pressure)
                                              / (k_param * RSF_param_L)), 0.2);
              }

            min_state_weakening_time_step = std::min(min_state_weakening_time_step,
                                                     delta_theta_max * RSF_param_L / V_p);

            /*
                        max_local_speed_over_meshsize = std::max(max_local_speed_over_meshsize,
                                                                 max_local_velocity
                                                                 /
                                                                 cell->minimum_vertex_distance());*/

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
                                        "with rate and state friction. min dt computed as follows: .... ")
  }
}

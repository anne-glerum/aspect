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
      /**
       * this time stepping plugin is developed to capture all time scales
       * relevant for rate-and-state friction seismic cycle models. For its
       * development see \cite{lapusta_elastodynamic_2000},
       * \cite{lapusta_three-dimensional_2009} and \cite{herrendorfer_invariant_2018}.
       */
      const MaterialModel::ViscoPlastic<dim> &viscoplastic
        = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());

      const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                               this->get_parameters().stokes_velocity_degree);

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values |
                               update_gradients |
                               update_quadrature_points);

      const unsigned int n_q_points = quadrature_formula.size();

      // get the velocities and "in" and "out"
      std::vector<Tensor<1,dim> > velocity_values(n_q_points);
      MaterialModel::MaterialModelInputs<dim> in(n_q_points,
                                                 this->introspection().n_compositional_fields);
      MaterialModel::MaterialModelOutputs<dim> out(n_q_points,
                                                   this->introspection().n_compositional_fields);

      // The Lapusta adaptive time stepping is based on four individual minimum time step criteria
      double min_state_weakening_time_step =  std::numeric_limits<double>::max();
      double min_healing_time_step =  std::numeric_limits<double>::max();
      double min_displacement_time_step =  std::numeric_limits<double>::max();
      double min_vep_relaxation_time_step =  std::numeric_limits<double>::max();

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            in.reinit(fe_values,
                      cell,
                      this->introspection(),
                      this->get_solution());

            this->get_material_model().evaluate(in, out);

            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);
            double max_local_velocity = 0;
            const double delta_x = cell->minimum_vertex_distance();
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                // TODO in Lapusta, this should be plastic velocity. But just taking the full velocity
                // should be the most conservative approach, so it should be ok...
                const double local_velocity = velocity_values[q].norm();

                std::pair<double,double> delta_theta_max_and_critical_slip_distance = viscoplastic.compute_delta_theta_max(
                                                                                        in.composition[0],  // should this be 0 or q? for all the times I hand over composition.. found 0 in some part of the code, but q intuitively makes more sense
                                                                                        in.position[q],
                                                                                        delta_x,
                                                                                        in.pressure[q]);

                min_state_weakening_time_step = std::min (min_state_weakening_time_step,
                                                          delta_theta_max_and_critical_slip_distance.first
                                                          * delta_theta_max_and_critical_slip_distance.second / local_velocity);

                // state healing time step is: Deltat_h = 0.2 * theta.
                min_healing_time_step = std::min (min_healing_time_step,
                                                  viscoplastic.compute_min_healing_time_step(in.composition[q]));

                // the maximum local velocity needed for the displacement time step
                max_local_velocity = std::max (max_local_velocity,
                                               local_velocity);

                // the viscoelastoplastic relaxation time step using the relaxation time scale: f_max * viscoplastic viscosity / shear modulus
                // with f_max = 0.2 in Herrendörfer et al. 2018
                // to capture the increasing slip rate in case of a purely rate-dependent friction, i.e. if b = 0
                min_vep_relaxation_time_step = std::min (min_vep_relaxation_time_step,
                                                         0.2 * out.viscosities[q]
                                                         / viscoplastic.get_elastic_shear_modulus(in.composition[0]));
              }

            // minimum displacement time step: Delta t_d = Delta d_max * min(|Delta x/v_x|,|Delta x/v_y|),
            // with Delta d_max = 1e-3 in Herrendörfer et al. 2018
            // here, the term  min(|Delta x/v_x|,|Delta x/v_y|) is simplified to min Delta x / max_local_velocity
            min_displacement_time_step = std::min (min_displacement_time_step,
                                                   1.e-3 * cell->minimum_vertex_distance() / max_local_velocity);

          }

      double min_lapusta_timestep = std::numeric_limits<double>::max();

      // take the minimum of the four criteria
      // TODO: is there a more elegant way to take the minimum of four values?
      min_lapusta_timestep = std::min (min_state_weakening_time_step,
                                       std::min (min_healing_time_step,
                                                 std::min (min_displacement_time_step,
                                                           min_vep_relaxation_time_step)));

      AssertThrow (min_state_weakening_time_step > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length was: " + std::to_string(min_lapusta_timestep) + ". "
                              "Please check for non-positive material properties."));

      AssertThrow (min_healing_time_step > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length was: " + std::to_string(min_lapusta_timestep) + ". "
                              "Please check for non-positive material properties."));

      AssertThrow (min_displacement_time_step > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length was: " + std::to_string(min_lapusta_timestep) + ". "
                              "Please check for non-positive material properties."));

      AssertThrow (min_vep_relaxation_time_step > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length was: " + std::to_string(min_lapusta_timestep) + ". "
                              "Please check for non-positive material properties."));

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
                                        "This time stepping plugin is developed to capture all time scales "
                                        "relevant for rate-and-state friction seismic cycle models. "
                                        "It is following the approach described in \\cite{lapusta_elastodynamic_2000}, "
                                        "\\cite{lapusta_three-dimensional_2009} and \\cite{herrendorfer_invariant_2018}. "
                                        "This time stepping plugin will enforce the use of the smallest of the four "
                                        "proposed time scales in the above papers. These are: the state weakening time step, "
                                        "the healing time step, the displacement time step, and the "
                                        "viscoelastoplastic relaxation time step. ")
  }
}

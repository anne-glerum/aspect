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

      // The Lapusta adaptive time stepping in \cite{herrendorfer_invariant_2018} is based on four individual minimum time step criteria:
      // state weakening timestep,
      // healing timestep,
      // displacement timestep,
      // vep relaxation timestep
      std::vector<double> timestep(4,std::numeric_limits<double>::max());

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        {
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

      const double delta_x = cell->minimum_vertex_distance();

              std::vector<double> timestep_for_this_cell = compute_lapusta_timestep_components(delta_x,
                                                           in,
                                                           out,
                                                           n_q_points,
                                                           velocity_values);
              for (unsigned int c=0; c<4; ++c)
                timestep[c] = std::min(timestep[c], timestep_for_this_cell[c]);
            }
        }

      // take the minimum of the four criteria
      double min_lapusta_timestep = *min_element(timestep.begin(), timestep.end());

      // communicate the min lapusta timestep between the processes
      const double min_global_lapusta_timestep
        = Utilities::MPI::min (min_lapusta_timestep, this->get_mpi_communicator());

      // ToDo ? In \cite{herrendorfer_invariant_2018 they multiply the min lapusta time step with a
      // "time step factor". Its size is investigated in their section 4 and in table 1
      // they set it to 1.0. Hence, there is no point in making a new variable for it,
      // but to have the same approach as them and be able to modify it, we could do
      // this:
      // min_global_lapusta_timestep = min_global_lapusta_timestep*timestepfactor;

      AssertThrow (min_lapusta_timestep > 0,
                   ExcMessage("The time step length for the each time step needs to be positive, "
                              "but the computed step length of the overall Lapusta time step "
                              "was: " + std::to_string(min_lapusta_timestep) + ". \n"
                              "The computed step length of the min state weakening time step "
                              "was: " + std::to_string(timestep[0]) + ". \n"
                              "The computed step length of the min healing time step "
                              "was: " + std::to_string(timestep[1]) + ". \n"
                              "The computed step length of the min displacement time step "
                              "was: " + std::to_string(timestep[2]) + ". \n"
                              "The computed step length of the min vep relaxation time step "
                              "was: " + std::to_string(timestep[3]) + ". \n"
                              "Please check for non-positive material properties."));

      /* ToDo: this needs to go into statistics!!!
      // Print lapusta timestep lengths
      // ToDo: only print this once and not for all processors! Or better print it once within the statistics file
      const char *unit = ( SimulatorAccess<dim>::convert_output_to_years() ? "years" : "seconds");
      const double multiplier = ( SimulatorAccess<dim>::convert_output_to_years() ? 1./year_in_seconds : 1.0);
      std::cout << "   Lapusta timestep length determined for next timestep: " ;
      std::cout << min_lapusta_timestep *multiplier << ' ' << unit << std::endl;
      std::cout << "      The components are:" << std::endl;
      std::cout << "      - the min state weakening time step: " ;
      std::cout << min_state_weakening_time_step *multiplier << ' ' << unit << std::endl;
      std::cout << "      - the min healing time step: ";
      std::cout << min_healing_time_step *multiplier << ' ' << unit << std::endl;
      std::cout << "      - the min displacement time step: ";
      std::cout << min_displacement_time_step *multiplier << ' ' << unit << std::endl;
      std::cout << "      - the min vep relaxation time step: ";
      std::cout << min_vep_relaxation_time_step *multiplier << ' ' << unit << std::endl << std::endl; */

      return min_global_lapusta_timestep;
    }




    template <int dim>
    std::vector<double>
    LapustaTimeStep<dim>::compute_lapusta_timestep_components(const double delta_x,
                                                              MaterialModel::MaterialModelInputs<dim> &in,
                                                              MaterialModel::MaterialModelOutputs<dim> &out,
                                                              const unsigned int n_q_points,
                                                              std::vector<Tensor<1,dim> > velocity_values) const
    {
      const MaterialModel::ViscoPlastic<dim> &viscoplastic
        = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());

      double max_local_velocity = 0;

      double min_state_weakening_time_step = std::numeric_limits<double>::max();
      double min_healing_time_step = std::numeric_limits<double>::max();
      double min_vep_relaxation_time_step = std::numeric_limits<double>::max();

      // Lapusta timestep is only relevant for the processes within the fault (RSF material), so dont compute it outside
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          if (in.composition[q][this->introspection().compositional_index_for_name("fault")] > 0.5)
            {
              // TODO in Lapusta, this should be plastic velocity. But just taking the full velocity
              // should be the most conservative approach, so it should be ok...
              const double local_velocity = velocity_values[q].norm();

              if (local_velocity >0)
                {
                  std::pair<double,double> delta_theta_max_and_critical_slip_distance = viscoplastic.compute_delta_theta_max(
                                                                                          in.position[q],
                                                                                          delta_x,
                                                                                          in.pressure[q]);

                  min_state_weakening_time_step = std::min (min_state_weakening_time_step,
                                                            delta_theta_max_and_critical_slip_distance.first
                                                            * delta_theta_max_and_critical_slip_distance.second
                                                            / local_velocity);

                  // the maximum local velocity needed for the displacement time step
                  max_local_velocity = std::max (max_local_velocity,
                                                 local_velocity);
                }
              // state healing time step is: Deltat_h = 0.2 * theta.
              min_healing_time_step = std::min (min_healing_time_step,
                                                viscoplastic.compute_min_healing_time_step(in.composition[q]));

              // the viscoelastoplastic relaxation time step using the relaxation time scale:
              // f_max * viscoplastic viscosity / shear modulus
              // with f_max = 0.2 in Herrendörfer et al. 2018
              // to capture the increasing slip rate in case of a purely rate-dependent friction, i.e. if b = 0
              min_vep_relaxation_time_step = std::min (min_vep_relaxation_time_step,
                                                       0.2 * out.viscosities[q]
                                                       / viscoplastic.get_elastic_shear_modulus(in.composition[q]));
              /*std::cout<< "the current vep relaxation time step is: "<< 0.2 * out.viscosities[q] / viscoplastic.get_elastic_shear_modulus(in.composition[q]) <<std::endl;
              std::cout<< "the new vep relaxation time step is: "<< min_vep_relaxation_time_step <<std::endl;
              std::cout<< "the shear modulus is: "<<viscoplastic.get_elastic_shear_modulus(in.composition[q]) << " and the viscosity is: "<<  out.viscosities[q] << std::endl<<std::endl;*/
            }
        }

      // minimum displacement time step: Delta t_d = Delta d_max * min(|Delta x/v_x|,|Delta x/v_y|),
      // with Delta d_max = 1e-3 in Herrendörfer et al. 2018
      // here, the term  min(|Delta x/v_x|,|Delta x/v_y|) is simplified to min(|Delta x / max_local_velocity|)
      // ToDo: check how different the min displacement timestep is from the convection timestep. Do
      // we need them both? Should it also be constraint to within the fault or used everywhere?
      const double min_displacement_time_step = std::min(1.e-3 * delta_x / max_local_velocity,std::numeric_limits<double>::max());

      return {min_state_weakening_time_step, min_healing_time_step, min_displacement_time_step, min_vep_relaxation_time_step};
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

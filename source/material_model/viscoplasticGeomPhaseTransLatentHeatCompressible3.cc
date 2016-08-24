/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/
/*  $Id: Clinvisc.cc 1430 2013-10-27 10:10 glerum $  */


#include <aspect/material_model/viscoplasticGeomPhaseTransLatentHeatCompressible3.h>
#include <deal.II/base/parameter_handler.h>


using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {

////////////////////////////////////////////////////////////////////////////////////////////////////
// In this template we calculate the phase transition function value using the non-dimensional equation (21) from Ita & King (1994) 
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    phase_transition_function (const double temperature,
                               const int j,
                               const Point<dim> &position) const
    {
      // The phase trans temperature is calculated as the adiabatic temperature
      double transition_temperature = pTsm*exp(thermal_alpha*9.81*(transition_depths[j])/reference_specific_heat);

      double phase_trans_func = 0.5*(1.0 + tanh(((height_to_surface-position[1]) - transition_depths[j] - Clapeyron_slopes[j] * (transition_depths[j] / transition_pressures[j]) * (temperature - transition_temperature)) / transition_widths[j]));

      return phase_trans_func;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    phase_function_derivative (const double temperature,
                               const int j,
                               const double pressure,
                               const Point<dim> &position) const
    {
      // The phase trans temperature is calculated as the adiabatic temperature
      double transition_temperature 	= pTsm*exp(thermal_alpha*9.81*(transition_depths[j])/reference_specific_heat);
      
      double pressure_deviation 	= pressure - transition_pressures[j] - Clapeyron_slopes[j] * (temperature - transition_temperature);

      double pressure_width 		= transition_pressures[j] * transition_widths[j] / transition_depths[j];

      double phase_func_derivative 	= 0.5 / pressure_width * (1.0 - std::tanh(pressure_deviation / pressure_width) * std::tanh(pressure_deviation / pressure_width));
      
      return phase_func_derivative;
    }
////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * What is the field with the maximum value?
     */
    template <int dim>
    int
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    maximum_composition(const std::vector<double> &comp) const
    {
      int maximum_comp = 0;
      double max = comp[0];

      for (unsigned int i = 1; i<comp.size(); i++)
        {
          if (comp[i] > max)
            {
              maximum_comp = i;
              max = comp[i];
            }
        }

      return maximum_comp;

    }

    /*
     * What is the harmonic average?
     */
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    harmonic_average(const std::vector<double> &comp,
                     const std::vector<double> &eta) const
    {
      double visc = 0.0;
      double compo_total = 0.0;

      for (unsigned int i = 0; i<n_compositional_fields; ++i)
        {
          visc += std::max(std::min(comp[i],1.0),0.0) / eta[i];
          compo_total += std::max(std::min(comp[i],1.0),0.0);
        }

      visc /= compo_total;
      visc = 1.0 / visc;

      return visc;
    }

    /*
     * What is the arithmetic average?
     */
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    arithmetic_average(const std::vector<double> &comp,
                       const std::vector<double> &eta) const
    {
      double visc = 0.0;
      double compo_total = 0.0;

      for (unsigned int n=0; n<n_compositional_fields; n++)
        {
          visc += std::max(std::min(comp[n],1.0),0.0) * eta[n];
          compo_total += std::max(std::min(comp[n],1.0),0.0);
        }

      visc /= compo_total;
      return visc;
    }

    /*
     * What is the geometric average?
     */
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    geometric_average(const std::vector<double> &comp,
                      const std::vector<double> &eta) const
    {
      double visc = 0.0;
      double compo_total = 0.0;

      for (unsigned int n=0; n<n_compositional_fields; n++)
        {
          visc += std::max(std::min(comp[n],1.0),0.0) * log10(eta[n]);
          compo_total += std::max(std::min(comp[n],1.0),0.0);
        }

      visc /= compo_total;
      visc = pow(10,visc);

      return visc;
    }


    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    diffusion(const double prefactor,
              const double activation_energy,
              const double activation_volume,
              const double temperature,
              const double pressure,
              const double nu) const
    {
      const double R = 8.314;
      return 0.5 * nu * (1e0/prefactor)*exp((activation_energy+activation_volume*pressure)/(R*std::max(1.0,temperature)));
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    dislocation(const double prefactor,
                const double stress_exponent,
                const double activation_energy,
                const double activation_volume,
                const double temperature,
                const double pressure,
                const double strain_rate_norm,
                const double nu) const
    {
      const double R = 8.314;
      return (0.5 * nu * std::pow(prefactor,-1e0/stress_exponent)*
              std::pow(strain_rate_norm,(1e0-stress_exponent)/stress_exponent)*
              exp((activation_energy+activation_volume*pressure)/(stress_exponent*R*std::max(1.0,temperature))));
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    plastic(const double phi,
            const double cohesion,
            const double pressure,
            const double strain_rate_norm) const

    {
      double strength = 0;
      if (dim==3)
        {
          const int ct = -1;
          strength = ((6.0*cohesion*std::cos(phi))/(std::sqrt(3.0)*(3.0+ct*std::sin(phi)))) +
                     ((2.0*std::sin(phi))/(std::sqrt(3.0)*(3.0+ct*std::sin(phi)))) * std::max(pressure,0.0);
        }
      else strength = std::max(pressure,0.0) * std::sin(phi) + cohesion * std::cos(phi);
      return strength / (2.0*strain_rate_norm);
    }


    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    viscosity (const double temperature,
               const double pressure,
               const std::vector<double> &composition,
               const SymmetricTensor<2,dim> &strain_rate,
               const Point<dim> &position) const
    {
      double viscosity_plastic = 0.0;
      double viscosity_viscous = 0.0;
      double viscosity = 0.0;

      /****************************************FIRSTITERATION*******************************************/
      if (this->get_nonlinear_iteration_number() == 0 && this->get_timestep_number() == 0 && this->get_initial_amr_number() < initial_adaptive_refinement)
        {
          // Multiple compositions
          if (n_compositional_fields>0)
            {
              if (viscosity_averaging=="Max")
                {
                  int max_comp = maximum_composition(composition);
                  viscosity = init_eta_fields[max_comp];
                }
              else if (viscosity_averaging=="Harmonic")
                {
                  viscosity = harmonic_average(composition, init_eta_fields);
                }
              else
                {
                  viscosity = geometric_average(composition, init_eta_fields);
                }
            }
          // No additional compositional fields
          else
            {
              viscosity = initial_eta;
            }
          if (weak_zone)
            {
              viscosity /= weak_zone_function.value(position);
            }
          if (harmonic_max)
            {
              viscosity = 1.0 / ((1.0 / viscosity) + (1.0 / maximum_eta));
              viscosity += minimum_eta;
            }
          else
            {
              viscosity = std::max(std::min(viscosity,maximum_eta),minimum_eta);
            }

          return viscosity;
        }

      /**************************STRAINRATE****************************************************************************/
      // Calculate the second invariant of the deviatoric strain rate tensor
      const SymmetricTensor<2,dim> strain_rate_dev = deviator(strain_rate);
      const double strain_rate_dev_inv = std::sqrt(0.5) * strain_rate_dev.norm();

      /************************VISCOPLASTIC****************************************************************************/
      // Multiple compositional fields
      if (composition.size()>0)
        {
          if (viscosity_averaging=="Max")
            {
              int max_comp = maximum_composition(composition);
              // On the first grid, the crust isn't well resolved anyway. This speeds up computations.
/*              if (this->get_initial_amr_number() < initial_adaptive_refinement)
                {
                  if (max_comp==3)
                    {
                      max_comp=2;
                    }
                }
*/
              const double visc_diffusion_inverse = 1.0 / diffusion(prefactors_diffusion_fields[max_comp],
                                                                    activation_energies_diffusion_fields[max_comp],
                                                                    activation_volumes_diffusion_fields[max_comp],
                                                                    temperature,
                                                                    pressure,
                                                                    nu_diffusion_fields[max_comp]);

              const double visc_dislocation_inverse = 1.0 / dislocation(prefactors_dislocation_fields[max_comp],
                                                                        stress_exponents_fields[max_comp],
                                                                        activation_energies_dislocation_fields[max_comp],
                                                                        activation_volumes_dislocation_fields[max_comp],
                                                                        temperature,
                                                                        pressure,
                                                                        strain_rate_dev_inv,
                                                                        nu_dislocation_fields[max_comp]);

              viscosity_viscous = 1.0 / (visc_diffusion_inverse + visc_dislocation_inverse);

              viscosity_plastic =  plastic(phis_fields[max_comp],
                                           cohesions_fields[max_comp],
                                           pressure,
                                           strain_rate_dev_inv);

              if (strain_rate_weakening)
                {
                  viscosity_plastic *= (std::max(1.0-(strain_rate_dev_inv/ref_strain_rate),0.1));
                }

              if (harmonic_plastic_viscous)
                {
                  viscosity = 1.0 / ((1.0/viscosity_viscous) + (1.0/viscosity_plastic));
                }
              else
                {
                  viscosity = std::min(viscosity_viscous,viscosity_plastic);
                }

              if (weak_zone)
                {
                  viscosity /= weak_zone_function.value(position);
                }

              if (harmonic_max)
                {
                  viscosity = 1.0 / ((1.0 / viscosity) + (1.0 / maximum_eta));
                  viscosity += minimum_eta;
                }
              else
                {
                  viscosity = std::max(std::min(viscosity,maximum_eta),minimum_eta);
                }
            }

          // Harmonic and geometric averaging
          else
            {
              std::vector<double> visc_diffusion_inverse(n_compositional_fields);
              std::vector<double> visc_dislocation_inverse(n_compositional_fields);
              std::vector<double> visc_viscous(n_compositional_fields);
              std::vector<double> visc_plastic(n_compositional_fields);
              std::vector<double> visc_effective(n_compositional_fields);

              for (unsigned int n=0; n<composition.size(); n++)
                {
                  visc_diffusion_inverse[n] = 1.0 / diffusion(prefactors_diffusion_fields[n],
                                                              activation_energies_diffusion_fields[n],
                                                              activation_volumes_diffusion_fields[n],
                                                              temperature,
                                                              pressure,
                                                              nu_diffusion_fields[n]);

                  visc_dislocation_inverse[n] = 1.0 / dislocation(prefactors_dislocation_fields[n],
                                                                  stress_exponents_fields[n],
                                                                  activation_energies_dislocation_fields[n],
                                                                  activation_volumes_dislocation_fields[n],
                                                                  temperature,
                                                                  pressure,
                                                                  strain_rate_dev_inv,
                                                                  nu_dislocation_fields[n]);

                  visc_viscous[n] = 1.0 / (visc_diffusion_inverse[n] + visc_dislocation_inverse[n]);

                  visc_plastic[n] =  plastic(phis_fields[n],
                                             cohesions_fields[n],
                                             pressure,
                                             strain_rate_dev_inv);

                  if (strain_rate_weakening)
                    {
                      visc_plastic[n] *= (std::max(1.0-(strain_rate_dev_inv/ref_strain_rate),0.1));
                    }

                  if (harmonic_plastic_viscous)
                    {
                      visc_effective[n] = 1.0 / ((1.0/visc_plastic[n]) + (1.0/visc_viscous[n]));
                    }
                  else
                    {
                      visc_effective[n] = std::min(visc_viscous[n],visc_plastic[n]);
                    }
                }

              //Harmonic averaging of the viscosities of each composition
              if (viscosity_averaging == "Harmonic")
                {
                  viscosity = harmonic_average(composition, visc_effective);
                }
              else
                {
                  viscosity = geometric_average(composition,visc_effective);
                }

              if (weak_zone)
                {
                  viscosity /= weak_zone_function.value(position);
                }
              if (harmonic_max)
                {
                  viscosity = 1.0 / ((1.0 / viscosity) + (1.0 / maximum_eta));
                  viscosity += minimum_eta;
                }
              else
                {
                  viscosity = std::max(std::min(viscosity,maximum_eta),minimum_eta);
                }
            }

        }
      // No additional compositional fields
      else
        {

          const double visc_diffusion_inverse = 1.0/diffusion(prefactor_diffusion,
                                                              activation_energy_diffusion,
                                                              activation_volume_diffusion,
                                                              temperature,
                                                              pressure,
                                                              nu_diffusion);

          const double visc_dislocation_inverse = 1.0/dislocation(prefactor_dislocation,
                                                                  stress_exponent,
                                                                  activation_energy_dislocation,
                                                                  activation_volume_dislocation,
                                                                  temperature,
                                                                  pressure,
                                                                  strain_rate.norm(),
                                                                  nu_dislocation);

          viscosity_viscous = 1.0 / (visc_diffusion_inverse + visc_dislocation_inverse);

          viscosity_plastic = plastic(phi,
                                      C,
                                      pressure,
                                      strain_rate_dev_inv);

          if (strain_rate_weakening)
            {
              viscosity_plastic *= (std::max(1.0-(strain_rate_dev_inv/ref_strain_rate),0.1));
            }

          if (harmonic_plastic_viscous)
            {
              viscosity = 1.0 / ((1.0/viscosity_plastic) + (1.0/viscosity_viscous));
            }
          else
            {
              viscosity = std::min(viscosity_plastic,viscosity_viscous);
            }

          if (weak_zone)
            {
              viscosity /= weak_zone_function.value(position);
            }
          if (harmonic_max)
            {
              viscosity = 1.0 / ((1.0 / viscosity) + (1.0 / maximum_eta));
              viscosity += minimum_eta;
            }
          else
            {
              viscosity = std::max(std::min(viscosity,maximum_eta),minimum_eta);
            }
        }

      return viscosity;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    viscosity_ratio (const double temperature,
                     const double pressure,
                     const std::vector<double> &composition,
                     const SymmetricTensor<2,dim> &strain_rate,
                     const Point<dim> &position) const
    {

      double viscosity_plastic = 0.0;
      double viscosity_viscous = 0.0;
      double compo_total = 0.0;
      const SymmetricTensor<2,dim> strain_rate_dev = deviator(strain_rate);
      const double strain_rate_dev_inv = std::sqrt(0.5) * strain_rate_dev.norm();
      double ratio = 0.0, ratio_plastic = 0.0, ratio_composite = 0.0;

      if (composition.size() > 0)
        {
          if (viscosity_averaging == "Max")
            {
              const int max_comp = maximum_composition(composition);
              const double visc_dislocation = dislocation(prefactors_dislocation_fields[max_comp],
                                                          stress_exponents_fields[max_comp],
                                                          activation_energies_dislocation_fields[max_comp],
                                                          activation_volumes_dislocation_fields[max_comp],
                                                          temperature,
                                                          pressure,
                                                          strain_rate_dev_inv,
                                                          nu_dislocation_fields[max_comp]);
              const double visc_diffusion = diffusion(prefactors_diffusion_fields[max_comp],
                                                      activation_energies_diffusion_fields[max_comp],
                                                      activation_volumes_diffusion_fields[max_comp],
                                                      temperature,
                                                      pressure,
                                                      nu_diffusion_fields[max_comp]);
              const double visc_plastic = plastic(phis_fields[max_comp],
                                                  cohesions_fields[max_comp],
                                                  pressure,
                                                  strain_rate_dev_inv);

              ratio_composite = visc_dislocation / visc_diffusion;
              ratio_plastic = (1.0 / ((1.0/visc_dislocation) + (1.0/visc_diffusion))) / visc_plastic;

              if (ratio_plastic <= 1.0 && ratio_composite <= 1.0)
                {
                  ratio = 0.0; //dislocation = 0
                }
              else if (ratio_plastic <= 1.0 && ratio_composite > 1.0)
                {
                  ratio = -1.0; //diffusion = -1
                }
              else ratio = 1.0; //plasticity = 1
            }
          // Harmonic and geometric averaging
          else
            {
              std::vector<double> visc_diffusion(n_compositional_fields);
              std::vector<double> visc_dislocation(n_compositional_fields);
              std::vector<double> visc_viscous(n_compositional_fields);
              std::vector<double> visc_plastic(n_compositional_fields);
              for (unsigned int n = 0; n < composition.size(); ++n)
                {
                  visc_dislocation[n] = dislocation(prefactors_dislocation_fields[n],
                                                    stress_exponents_fields[n],
                                                    activation_energies_dislocation_fields[n],
                                                    activation_volumes_dislocation_fields[n],
                                                    temperature,
                                                    pressure,
                                                    strain_rate_dev_inv,
                                                    nu_dislocation_fields[n]);
                  visc_diffusion[n] = diffusion(prefactors_diffusion_fields[n],
                                                activation_energies_diffusion_fields[n],
                                                activation_volumes_diffusion_fields[n],
                                                temperature,
                                                pressure,
                                                nu_diffusion_fields[n]);
                  visc_viscous[n] = 1.0 / ((1.0 / visc_dislocation[n]) + (1.0 / visc_diffusion[n]));

                  visc_plastic[n] = plastic(phis_fields[n],
                                            cohesions_fields[n],
                                            pressure,
                                            strain_rate_dev_inv);
                }

              double viscosity_dislocation = 0.0, viscosity_diffusion = 0.0;

              //Harmonic averaging of the viscosities of each composition
              if (viscosity_averaging == "Harmonic")
                {
                  viscosity_plastic = harmonic_average(composition, visc_plastic);
                  viscosity_dislocation = harmonic_average(composition, visc_dislocation);
                  viscosity_diffusion = harmonic_average(composition, visc_diffusion);
                  viscosity_viscous = harmonic_average(composition, visc_viscous);
                }
              // Geometric averaging
              else
                {
                  viscosity_plastic = geometric_average(composition, visc_plastic);
                  viscosity_dislocation = geometric_average(composition, visc_dislocation);
                  viscosity_diffusion = geometric_average(composition, visc_diffusion);
                  viscosity_viscous = geometric_average(composition, visc_viscous);
                }

              ratio_plastic = viscosity_viscous / viscosity_plastic;
              ratio_composite = viscosity_dislocation / viscosity_diffusion;
              if (ratio_plastic <= 1.0 && ratio_composite <= 1.0)
                {
                  ratio = 0.0; //dislocation = 0
                }
              else if (ratio <= 1.0 && ratio_composite > 1.0)
                {
                  ratio = -1.0; //diffusion = -1
                }
              else ratio = 1.0; //plasticity = 1
            }
        }

      else
        {
          const double visc_dislocation = dislocation(prefactor_dislocation,
                                                      stress_exponent,
                                                      activation_energy_dislocation,
                                                      activation_volume_dislocation,
                                                      temperature,
                                                      pressure,
                                                      strain_rate_dev_inv,
                                                      nu_dislocation);
          const double visc_diffusion = diffusion(prefactor_diffusion,
                                                  activation_energy_diffusion,
                                                  activation_volume_diffusion,
                                                  temperature,
                                                  pressure,
                                                  nu_diffusion);
          viscosity_plastic = plastic(phi,
                                      C,
                                      pressure,
                                      strain_rate_dev_inv);

          viscosity_viscous = 1.0 / ((1.0 / visc_dislocation) + (1.0 / visc_diffusion));
          ratio_plastic = viscosity_viscous / viscosity_plastic;
          ratio_composite = visc_dislocation / visc_diffusion;
          if (ratio_plastic <= 1.0 & ratio_composite <= 1.0)
            {
              ratio = 0.0; //dislocation = 0
            }
          else if (ratio <= 1.0 & ratio_composite > 1.0)
            {
              ratio = -1.0; //diffusion = -1
            }
          else ratio = 1.0; //plasticity = 1
        }
      return ratio;
    }




    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reference_viscosity () const
    {
      return reference_eta;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reference_density () const
    {
      return reference_rho;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reference_thermal_expansion_coefficient () const
    {
      return thermal_alpha;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    specific_heat (const double,
                   const double,
                   const std::vector<double> &composition, /*composition*/
                   const Point<dim> &) const
    {
      double cp_total=0.0;
      if (composition.size()>0)
        {
          if (viscosity_averaging == "Max")
            {
              const int max_comp = maximum_composition(composition);
              cp_total = capacities_fields[max_comp];
            }

          else
            {
              cp_total = arithmetic_average(composition, capacities_fields);
            }

        }
      else
        {
          cp_total = reference_specific_heat;
        }
      return cp_total;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reference_cp () const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    thermal_conductivity (const double,
                          const double,
                          const std::vector<double> &composition, /*composition*/
                          const Point<dim> &) const
    {
      double k_total=0.0;
      if (composition.size()>0)
        {
          //loop over all compositions //Max rule averaging
          if (viscosity_averaging == "Max")
            {
              const int max_comp = maximum_composition(composition);
              k_total = conductivities_fields[max_comp];
            }
          //loop over all compositions //arithmetic averaging
          else
            {
              k_total = arithmetic_average(composition, conductivities_fields);
            }
        }
      else
        {
          k_total = k_value;
        }
      return k_total;
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reference_thermal_diffusivity () const
    {
      return k_value/(reference_rho*reference_specific_heat);
    }

    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    density (const double temperature,
             const double pressure, 
             const std::vector<double> &composition, /*composition*/
             const Point<dim> &position) const
    {
      double refrho_total=0.0;
      double refT_total=0.0;

      if (composition.size()>0)
        {
          if (viscosity_averaging=="Max")
            {
              const int max_comp = maximum_composition(composition);
              refrho_total = refdens_fields[max_comp];
              refT_total = reftemps_fields[max_comp];
            }
          else
            {
              refrho_total = arithmetic_average(composition, refdens_fields);
              refT_total   = arithmetic_average(composition, reftemps_fields);
            }
        }
      else
        {
          refrho_total = reference_rho;
          refT_total   = reference_T;
        }

////////////////////////////////////////////////////////////////////////////////////////////////////
      const double kappa = compressibility(temperature,pressure,composition,position);
      const double pressure_dependence = refrho_total * kappa * (pressure - this->get_surface_pressure());

      refrho_total = (refrho_total + pressure_dependence) * (1.0 - thermal_alpha * (temperature - refT_total));

      return refrho_total;
////////////////////////////////////////////////////////////////////////////////////////////////////

//      return (refrho_total * (1.0 - thermal_alpha * (temperature - refT_total)));
    }


    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    thermal_expansion_coefficient (const double temperature,
                                   const double,
                                   const std::vector<double> &, /*composition*/
                                   const Point<dim> &) const
    {
      return thermal_alpha;
    }


    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    compressibility (const double,
                     const double,
                     const std::vector<double> &, /*composition*/
                     const Point<dim> &) const
    {
      return reference_compressibility;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    entropy_derivative (const double temperature,
                        const double pressure,
                        const std::vector<double> &compositional_fields,
                        const Point<dim> &position,
                        const NonlinearDependence::Dependence dependence) const
    {
      double entropy_derivative = 0.0;
      if(this->get_timestep_number() == 0){
	entropy_derivative = 0.0;}
      else{
      const double rho = density (temperature, pressure, compositional_fields, position);
      unsigned int number_of_phase_transitions = Clapeyron_slopes.size();

      if (this->include_latent_heat())
        for (unsigned int phase_trans=0; phase_trans<number_of_phase_transitions; ++phase_trans)
          {
            const double phase_func_derivative = phase_function_derivative(temperature, phase_trans, pressure, position);

            // calculate the change of entropy across the phase transition
            double deltaS = 0.0;
            if (phase_trans == 0)     	// comp[0] -> comp[3], comp[1] -> comp[5]
              deltaS = Clapeyron_slopes[phase_trans] * (refdens_fields[3] - refdens_fields[0]) / (rho * rho) * (compositional_fields[3]+compositional_fields[5]);
            else if (phase_trans == 1) 	// comp[3] -> comp[4], comp[5] -> comp[6]
              deltaS = Clapeyron_slopes[phase_trans] * (refdens_fields[4] - refdens_fields[3]) / (rho * rho) * (compositional_fields[4]+compositional_fields[6]);
            
	    // we need deltaS * dpi/dP for the pressure derivative and - deltaS * dpi/dP * gamma for the temperature derivative
            if (dependence == NonlinearDependence::pressure)
              entropy_derivative += phase_func_derivative * deltaS;
            else if (dependence == NonlinearDependence::temperature)
              entropy_derivative -= phase_func_derivative * deltaS * Clapeyron_slopes[phase_trans];
            else
              AssertThrow(false, ExcMessage("Error in calculating the entropy gradient"));
          }
      }
      return entropy_derivative;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    template <int dim>
    double
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    reaction_term (const double temperature,
		   const double pressure,
                   const std::vector<double> &compositional_fields,
                   const Point<dim> &position,
                   const unsigned int compositional_variable) const
    {
      double delta_C = 0.0;
      double phase_trans_func0 = phase_transition_function(temperature, 0, position);
      double phase_trans_func1 = phase_transition_function(temperature, 1, position);

      // Calculate the total compositional value of a 'transition-group' 
      double sumslab = compositional_fields[1]+compositional_fields[5]+compositional_fields[6];
      double summantle = compositional_fields[0]+compositional_fields[3]+compositional_fields[4];

      switch (compositional_variable)
      {
      case 0:
        delta_C = (1. - phase_trans_func0)*summantle - compositional_fields[0];
        break;
      case 1:
        delta_C = (1. - phase_trans_func0)*sumslab - compositional_fields[1];
        break;
      case 2:
        delta_C = 0.0;
        break;
      case 3:
        delta_C = (phase_trans_func0-phase_trans_func1)*summantle - compositional_fields[3];
        break;
      case 4:
        delta_C = (phase_trans_func1)*summantle - compositional_fields[4];
        break; 
      case 5:
        delta_C = (phase_trans_func0-phase_trans_func1)*sumslab - compositional_fields[5];
	break;
      case 6:
        delta_C = (phase_trans_func1)*sumslab - compositional_fields[6];
	break;
      default:
        delta_C = 0.0;
        break;
      }

      return delta_C;
    }
////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    viscosity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      if (n_compositional_fields != 0)
        {
          return ((dependence & NonlinearDependence::pressure)
                  ||
                  (dependence & NonlinearDependence::strain_rate)
                  ||
                  (dependence & NonlinearDependence::temperature)
                  ||
                  (dependence & NonlinearDependence::compositional_fields));
        }

      else
        {
          return ((dependence & NonlinearDependence::pressure)
                  ||
                  (dependence & NonlinearDependence::strain_rate)
                  ||
                  (dependence & NonlinearDependence::temperature));
        }
    }


    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    density_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none)
          &&
          (n_compositional_fields != 0))
        return true;
      else if (((dependence & NonlinearDependence::temperature) != NonlinearDependence::none)
               &&
               (thermal_alpha != 0))
        return true;
      else
        return false;

    }

    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    compressibility_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    specific_heat_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none)
          &&
          (n_compositional_fields != 0))
        return true;
      else
        return false;
    }

    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none)
          &&
          (n_compositional_fields != 0))
        return true;
      else
        return false;
    }


    template <int dim>
    bool
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::
    is_compressible () const
    {
      return false;
    }



    template <int dim>
    void
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Compositional fields");
      {
        prm.declare_entry ("Number of fields", "0",
                           Patterns::Integer (0),
                           "The number of fields that will be advected along with the flow field, excluding "
                           "velocity, pressure and temperature.");
        prm.declare_entry ("List of phis of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of angles of internal friction equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of conductivities of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of thermal conductivities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of capacities of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of heat capacities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of reftemps of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of reference temperatures equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of refdens of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of reference densities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of initial viscs of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of initial viscosities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of cohesions of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of cohesions equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of prefactors diffusion of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of prefactors of diffusion equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of activation energies diffusion of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of activation energies of diffusion equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of activation volumes diffusion of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of activation volumes of diffusion equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of constant coefficients nu diffusion of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of constant coefficients of diffusion equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of prefactors dislocation of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of prefactors of dislocation equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of constant coefficients nu dislocation of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of constant coefficients of dislocation equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of activation energies dislocation of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of activation energies of dislocation equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of activation volumes dislocation of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of activation volumes of dislocation equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of stress exponents of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of stress exponents equal to the number of "
                           "compositional fields.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Viscoplastic model");
        {
          prm.declare_entry ("Reference density", "3300",
                             Patterns::Double (0),
                             "Reference density $\\rho_0$ (for normalization). Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. Units: $K$.");
          prm.declare_entry ("Reference viscosity", "5e24",
                             Patterns::Double (0),
                             "The value of the reference constant viscosity (for normalization). Units: $kg/m/s$.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the reference thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the reference specific heat $cp$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");

////////////////////////////////////////////////////////////////////////////////////////////////////
          prm.declare_entry ("Reference compressibility", "5.124e-12",
                             Patterns::Double (0),
                             "Compressibility "
                             "Units: $1/Pa$.");
          prm.enter_subsection("Phase transitions");
          {
            prm.declare_entry ("List of transition depths", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
            prm.declare_entry ("List of Clapeyron slopes", "",
                           Patterns::List (Patterns::Double()),
                           "");
            prm.declare_entry ("List of transition widths", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
            prm.declare_entry ("List of transition pressures", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
          }
          prm.leave_subsection();
////////////////////////////////////////////////////////////////////////////////////////////////////

          prm.enter_subsection ("Viscosity");
          {
            prm.declare_entry ("Viscosity Averaging", "Harmonic",
                               Patterns::Anything (),
                               "Averaging of compositional field contributions to viscosity, "
                               "density, specific heat and thermal conductivity.");
            prm.declare_entry ("Harmonic viscous and plastic viscosity averaging", "true",
                               Patterns::Bool (),
                               "Averaging of viscous and plastic viscosity. Can be harmonic or minimum.");
            prm.declare_entry ("Harmonic effective and maximum viscosity averaging", "false",
                               Patterns::Bool (),
                               "Averaging of effective and maximum viscosity. Can be min/max or harmonic "
                               "max viscosity + adding "
                               "the minimum viscosity. The latter option is more smooth but can give "
                               "numerical breakdown.");
            prm.declare_entry ("Initial Viscosity", "5e22",
                               Patterns::Double (0),
                               "The value of the initial viscosity. Units: $kg/m/s$.");
            prm.declare_entry ("Minimum Viscosity", "1e20",
                               Patterns::Double (0),
                               "The value of the minimum viscosity cutoff. Units: $kg/m/s$.");
            prm.declare_entry ("Maximum Viscosity", "1e25",
                               Patterns::Double (0),
                               "The value of the maximum viscosity cutoff. Units: $kg/m/s$.");
            prm.declare_entry ("Weak zone", "false",
                               Patterns::Bool (),
                               "Presence of a weak zone.");
            Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
            prm.declare_entry ("Activation energy diffusion", "335e3",
                               Patterns::Double (0),
                               "Activation energy for diffusion creep");
            prm.declare_entry ("Activation volume diffusion", "4.0e-6",
                               Patterns::Double (0),
                               "Activation volume for diffusion creep");
            prm.declare_entry ("Prefactor diffusion", "1.92e-11",
                               Patterns::Double (0),
                               "Prefactor for diffusion creep "
                               "(1e0/prefactor)*exp((activation_energy+activation_volume*pressure)/(R*temperature))");
            prm.declare_entry ("Constant coefficient diffusion", "1.0",
                               Patterns::Double (0),
                               "Constant coefficient for diffusion creep");
            prm.declare_entry ("Activation energy dislocation", "540e3",
                               Patterns::Double (0),
                               "Activation energy for dislocation creep");
            prm.declare_entry ("Activation volume dislocation", "14.0e-6",
                               Patterns::Double (0),
                               "Activation volume for dislocation creep");
            prm.declare_entry ("Prefactor dislocation", "2.42e-10",
                               Patterns::Double (0),
                               "Prefactor for dislocation creep "
                               "(1e0/prefactor)*exp((activation_energy+activation_volume*pressure)/(R*temperature))");
            prm.declare_entry ("Stress exponent", "3.5",
                               Patterns::Double (0),
                               "Stress exponent for dislocation creep");
            prm.declare_entry ("Constant coefficient dislocation", "1.0",
                               Patterns::Double (0),
                               "Constant coefficient for dislocation creep");
            prm.declare_entry ("Angle internal friction", "20",
                               Patterns::Double (0),
                               "Angle of internal friction for plastic creep in absence of compositional fields");
            prm.declare_entry ("Cohesion", "20.0e6",
                               Patterns::Double (0),
                               "Cohesion for plastic creep in absence of compositional fields");
            prm.declare_entry ("Plastic strain rate weakening", "false",
                               Patterns::Bool (),
                               "Whether or not to use strain rate weakening in the calculation of plastic viscosity");
            prm.declare_entry ("Reference strain rate", "1e-15",
                               Patterns::Double (0),
                               "The reference strain rate used in strain rate weakening");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

////////////////////////////////////////////////////////////////////////////////////////////////////
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Subduction temperature2");
        {
          prm.declare_entry ("Height to surface", "660.e3",
                             Patterns::Double (0),
                             "");
          prm.declare_entry ("Potential temperature mantle on surface", "1600.",
                             Patterns::Double (0),
                             "");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
////////////////////////////////////////////////////////////////////////////////////////////////////

    }


    template <int dim>
    void
    ViscoplasticGeomPhaseTransLatentHeatCompressible3<dim>::parse_parameters (ParameterHandler &prm)
    {


      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Viscoplastic model");
        {

          reference_rho              = prm.get_double ("Reference density");
          reference_T                = prm.get_double ("Reference temperature");
          reference_eta              = prm.get_double ("Reference viscosity");
          k_value                    = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");

////////////////////////////////////////////////////////////////////////////////////////////////////
          reference_compressibility     = prm.get_double ("Reference compressibility");
          prm.enter_subsection("Phase transitions");
          {
            transition_depths             = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition depths")));

            Clapeyron_slopes              = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of Clapeyron slopes")));

            transition_widths             = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition widths")));

            transition_pressures          = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition pressures")));
          }
          prm.leave_subsection();
////////////////////////////////////////////////////////////////////////////////////////////////////

          prm.enter_subsection ("Viscosity");
          {
            viscosity_averaging      = prm.get ("Viscosity Averaging");
            AssertThrow(viscosity_averaging == "Max" || viscosity_averaging == "Harmonic" || viscosity_averaging == "Geometric",
                        ExcMessage("Invalid input parameter file: This type of viscosity averaging is not implemented"));
            harmonic_plastic_viscous = prm.get_bool ("Harmonic viscous and plastic viscosity averaging");
            harmonic_max      = prm.get_bool ("Harmonic effective and maximum viscosity averaging");
            initial_eta       = prm.get_double ("Initial Viscosity");
            minimum_eta       = prm.get_double ("Minimum Viscosity");
            maximum_eta       = prm.get_double ("Maximum Viscosity");
            weak_zone         = prm.get_bool   ("Weak zone");
            weak_zone_function.parse_parameters (prm);

            activation_energy_diffusion   = prm.get_double ("Activation energy diffusion");
            activation_volume_diffusion   = prm.get_double ("Activation volume diffusion");
            prefactor_diffusion           = prm.get_double ("Prefactor diffusion");
            nu_diffusion                  = prm.get_double ("Constant coefficient diffusion");
            activation_energy_dislocation = prm.get_double ("Activation energy dislocation");
            activation_volume_dislocation = prm.get_double ("Activation volume dislocation");
            prefactor_dislocation         = prm.get_double ("Prefactor dislocation");
            nu_dislocation                = prm.get_double ("Constant coefficient dislocation");
            stress_exponent               = prm.get_double ("Stress exponent");
            phi                           = prm.get_double ("Angle internal friction");
            C                             = prm.get_double ("Cohesion");
            strain_rate_weakening         = prm.get_bool ("Plastic strain rate weakening");
            ref_strain_rate               = prm.get_double ("Reference strain rate");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

////////////////////////////////////////////////////////////////////////////////////////////////////
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Subduction temperature");
        {
          pTsm                  = prm.get_double ("Potential temperature mantle on surface");
          height_to_surface     = prm.get_double ("Height to surface");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
////////////////////////////////////////////////////////////////////////////////////////////////////

      prm.enter_subsection ("Mesh refinement");
      {
        initial_adaptive_refinement = prm.get_integer ("Initial adaptive refinement");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Compositional fields");
      {
        n_compositional_fields = prm.get_integer ("Number of fields");

        if (n_compositional_fields > 0)
          {

            //parameters needed for all rheologies
            const std::vector<double> n_conductivities_fields = Utilities::string_to_double
                                                                (Utilities::split_string_list(prm.get ("List of conductivities of fields")));
            conductivities_fields = std::vector<double> (n_conductivities_fields.begin(),
                                                         n_conductivities_fields.end());
            AssertThrow (conductivities_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of conductivities of fields"));


            const std::vector<double> n_capacities_fields = Utilities::string_to_double
                                                            (Utilities::split_string_list(prm.get ("List of capacities of fields")));
            capacities_fields = std::vector<double> (n_capacities_fields.begin(),
                                                     n_capacities_fields.end());
            AssertThrow (capacities_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of capacities of fields"));

            const std::vector<double> n_reftemps_fields = Utilities::string_to_double
                                                          (Utilities::split_string_list(prm.get ("List of reftemps of fields")));
            reftemps_fields = std::vector<double> (n_reftemps_fields.begin(),
                                                   n_reftemps_fields.end());
            AssertThrow (reftemps_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of reftemps of fields"));


            const std::vector<double> n_refdens_fields = Utilities::string_to_double
                                                         (Utilities::split_string_list(prm.get ("List of refdens of fields")));
            refdens_fields = std::vector<double> (n_refdens_fields.begin(),
                                                  n_refdens_fields.end());
            AssertThrow (refdens_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of refdens of fields"));

            const std::vector<double> n_init_eta_fields = Utilities::string_to_double
                                                          (Utilities::split_string_list(prm.get ("List of initial viscs of fields")));
            init_eta_fields = std::vector<double> (n_init_eta_fields.begin(),
                                                   n_init_eta_fields.end());
            AssertThrow (init_eta_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of initial viscs of fields"));

            //plastic parameters

            const std::vector<double> n_cohesions_fields = Utilities::string_to_double
                                                           (Utilities::split_string_list(prm.get ("List of cohesions of fields")));
            cohesions_fields = std::vector<double> (n_cohesions_fields.begin(),
                                                    n_cohesions_fields.end());
            AssertThrow (cohesions_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of cohesions of fields"));

            const std::vector<double> n_phis_fields = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of phis of fields")));
            for (unsigned int i=0; i<n_compositional_fields; i++)
              {
                phis_fields.push_back(n_phis_fields[i]*numbers::PI/180.0);
              }
            AssertThrow (phis_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of phis of fields"));



            //Dislocation parameters


            const std::vector<double> n_stressexponent_fields = Utilities::string_to_double
                                                                (Utilities::split_string_list(prm.get ("List of stress exponents of fields")));
            stress_exponents_fields = std::vector<double> (n_stressexponent_fields.begin(),
                                                           n_stressexponent_fields.end());
            AssertThrow (stress_exponents_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of stress exponents of fields"));

            const std::vector<double> n_prefactordisl_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of prefactors dislocation of fields")));
            //correction of pure shear measurements for uniaxiality
            for (unsigned int i=0; i<n_compositional_fields; i++)
              {
                prefactors_dislocation_fields.push_back(n_prefactordisl_fields[i]*std::pow(3,(n_stressexponent_fields[i]+1.0)/2.0)*0.5);
              }
            AssertThrow (prefactors_dislocation_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of prefactors dislocation of fields"));

            const std::vector<double> n_actenergydisl_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of activation energies dislocation of fields")));
            activation_energies_dislocation_fields = std::vector<double> (n_actenergydisl_fields.begin(),
                                                                          n_actenergydisl_fields.end());
            AssertThrow (activation_energies_dislocation_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of activation energies dislocation of fields"));

            const std::vector<double> n_actvolumedisl_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of activation volumes dislocation of fields")));
            activation_volumes_dislocation_fields = std::vector<double> (n_actvolumedisl_fields.begin(),
                                                                         n_actvolumedisl_fields.end());
            AssertThrow (activation_volumes_dislocation_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of activation volumes dislocation of fields"));

            const std::vector<double> n_nu_fields = Utilities::string_to_double
                                                    (Utilities::split_string_list(prm.get ("List of constant coefficients nu dislocation of fields")));
            nu_dislocation_fields = std::vector<double> (n_nu_fields.begin(),
                                                         n_nu_fields.end());
            AssertThrow (nu_dislocation_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of constant coefficients nu dislocation of fields"));



            //diffusion parameters
            const std::vector<double> n_prefactordiff_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of prefactors diffusion of fields")));
            //correction for uniaxial measurements
            for (unsigned int i=0; i<n_compositional_fields; i++)
              {
                prefactors_diffusion_fields.push_back(n_prefactordiff_fields[i]*3.0*0.5);
              }
            AssertThrow (prefactors_diffusion_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of prefactors diffusion of fields"));

            const std::vector<double> n_actenergydiff_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of activation energies diffusion of fields")));
            activation_energies_diffusion_fields = std::vector<double> (n_actenergydiff_fields.begin(),
                                                                        n_actenergydiff_fields.end());
            AssertThrow (activation_energies_diffusion_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of activation energies diffusion of fields"));

            const std::vector<double> n_actvolumediff_fields = Utilities::string_to_double
                                                               (Utilities::split_string_list(prm.get ("List of activation volumes diffusion of fields")));
            activation_volumes_diffusion_fields = std::vector<double> (n_actvolumediff_fields.begin(),
                                                                       n_actvolumediff_fields.end());
            AssertThrow (activation_volumes_diffusion_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of activation volumes diffusion of fields"));

            const std::vector<double> n_nu_diff_fields = Utilities::string_to_double
                                                         (Utilities::split_string_list(prm.get ("List of constant coefficients nu diffusion of fields")));
            nu_diffusion_fields = std::vector<double> (n_nu_diff_fields.begin(),
                                                       n_nu_diff_fields.end());
            AssertThrow (nu_diffusion_fields.size() == n_compositional_fields,
                         ExcMessage("Invalid input parameter file: Wrong number of entries in List of constant coefficients nu diffusion of fields"));
          }
      }
      prm.leave_subsection();

    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(ViscoplasticGeomPhaseTransLatentHeatCompressible3,
                                   "viscoplasticGeomPhaseTransLatentHeatCompressible3",
                                   "A material model that allows for nonlinear rheologies for 1 or more compositions."
                                   "Viscosity can depend on temperature, pressure, strain rate and composition. "
                                   "Morh-Coulomb plasticity, dislocation creep and diffusion creep are implemented. "
                                   "Averaging of viscosity contributions occurs through harmonic, geometric or infinite norm "
                                   "rule averaging."
                                   "Density can depend on temperature and reference temperature and reference density, which depend on composition. "
                                   "Specific heat and thermal conductivity can depend on composition, but are otherwise constant. ")
  }
}

/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#include "multicomponent_vp.h"
#include "ascii_T_regions.h"
#include <aspect/simulator.h>
#include <deal.II/base/parameter_handler.h>

#include <numeric>

using namespace dealii;

namespace aspect
{
namespace MaterialModel
{

template <int dim>
const std::vector<double>
MulticomponentVP<dim>::
compute_volume_fractions( const std::vector<double> &compositional_fields) const
{
    std::vector<double> volume_fractions( compositional_fields.size()+1);
    //std::vector<double> volume_fractions( compositional_fields.size());

    //clip the compositional fields so they are between zero and one
    std::vector<double> x_comp = compositional_fields;
    for ( unsigned int i=0; i < x_comp.size(); ++i)
        x_comp[i] = std::min(std::max(x_comp[i], 0.0), 1.0);

    //sum the compositional fields for normalization purposes
    double sum_composition = 0.0;
    for ( unsigned int i=0; i < x_comp.size(); ++i)
        sum_composition += x_comp[i];

    if (sum_composition >= 1.0)
    {
        volume_fractions[0] = 0.0;  //background mantle
        for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1]/sum_composition;
        /*for ( unsigned int i=0; i <= x_comp.size(); ++i)
          volume_fractions[i] = x_comp[i]/sum_composition;*/
    }
    else
    {
        volume_fractions[0] = 1.0 - sum_composition; //background mantle
        for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1];
    }
    return volume_fractions;
}

template <int dim>
double
MulticomponentVP<dim>::
diffusion(const deformation def,
          const double temperature,
          const double pressure) const
{
    const double R = 8.314;
    return 0.5 * def.nu_diff * (1e0/def.A_diff)*exp((def.Q_diff+def.V_diff*pressure)/(R*std::max(1.0,temperature)));
}

template <int dim>
double
MulticomponentVP<dim>::
dislocation(const deformation def,
            const double temperature,
            const double pressure,
            const double strain_rate_norm) const
{
    const double R = 8.314;
    return (0.5 * def.nu_disl * std::pow(def.A_disl,-1e0/def.n)*
            std::pow(strain_rate_norm,(1e0-def.n)/def.n)*
            exp((def.Q_disl+def.V_disl*pressure)/(def.n*R*std::max(1.0,temperature))));
}

template <int dim>
double
MulticomponentVP<dim>::
Peierls(const deformation def,
        const double temperature,
        const double pressure,
        const double strain_rate_norm) const
{
    const double R = 8.314;
    return (0.5 * def.nu_Peierls * std::pow(def.A_Peierls,-1e0/def.n_Peierls)*
            std::pow(strain_rate_norm,(1e0-def.n_Peierls)/def.n_Peierls)*
            exp((def.Q_Peierls+def.V_Peierls*pressure)/(def.n_Peierls*R*std::max(1.0,temperature))));
}

template <int dim>
double
MulticomponentVP<dim>::
plastic(const deformation def,
        const double pressure,
        const double strain_rate_norm) const

{
    double strength = 0;
    if (dim==3)
    {
        const int ct = -1;
        strength = ((6.0*def.C*std::cos(def.phi))/(std::sqrt(3.0)*(3.0+ct*std::sin(def.phi)))) +
                   ((6.0*std::sin(def.phi))/(std::sqrt(3.0)*(3.0+ct*std::sin(def.phi)))) * std::max(pressure,0.0);
    }
    else strength = std::max(pressure,0.0) * std::sin(def.phi) + def.C * std::cos(def.phi);
    return strength / (2.0*strain_rate_norm);
}

template <int dim>
void
MulticomponentVP<dim>::
evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
    MaterialModel::MaterialModelOutputs<dim> &out) const
    {
  for (unsigned int i=0; i < in.position.size(); ++i)
  {
    out.viscosities[i] = viscosity (in.temperature[i],in.pressure[i],in.composition[i],in.strain_rate[i],in.position[i]);

    out.densities[i] = density (in.temperature[i],in.pressure[i],in.composition[i],in.position[i]);

    out.thermal_expansion_coefficients[i] = thermal_expansion_coefficient (in.temperature[i],in.pressure[i],in.composition[i],in.position[i]);
    out.specific_heat[i] = specific_heat (in.temperature[i],in.pressure[i],in.composition[i],in.position[i]);
    out.thermal_conductivities[i] = thermal_conductivity (in.temperature[i],in.pressure[i],in.composition[i],in.position[i]);
    out.compressibilities[i] = compressibility (in.temperature[i],in.pressure[i],in.composition[i],in.position[i]);
    // Pressure derivative of entropy at the given positions.
    out.entropy_derivative_pressure[i] = 0.0;
    // Temperature derivative of entropy at the given positions.
    out.entropy_derivative_temperature[i] = 0.0;
    // Change in composition due to chemical reactions at the
    // given positions. The term reaction_terms[i][c] is the
    // change in compositional field c at point i.
    for (unsigned int c=0; c<in.composition[i].size(); ++c)
      out.reaction_terms[i][c] = 0.0;
  }
    }

template <int dim>
double
MulticomponentVP<dim>::
viscosity (const double temperature,
           const double pressure,
           const std::vector<double> &composition,       /*composition*/
           const SymmetricTensor<2,dim> &strain_rate,
           const Point<dim> &position) const
{

    //const bool do_660_visc_smoothing = false;
    double viscosity = 0.0;

    std::vector<double> volume_fractions = compute_volume_fractions(composition);

    // To compensate for erroneous Aegean weak zone volume, we change the viscosity law for depths less than 120km
    if(do_660_visc_smoothing)
    {
    	deformation_types[0] = upper_mantle_deformation_type;
//                this->get_pcout() << "> 660 smoothing " << std::endl;
    }

    bool in_OOC = false;
//      if (InitialTemperature::AsciiPip<dim> *ic = dynamic_cast<InitialTemperature::AsciiPip<dim> *>
//          (const_cast<InitialTemperature::Interface<dim> *>(&this->get_initial_temperature())))
    if (this->get_initial_temperature_manager().template has_matching_initial_temperature_model<const InitialTemperature::AsciiPip<dim> >())
         {
           const InitialTemperature::AsciiPip<dim> &ic = this->get_initial_temperature_manager().template get_matching_initial_temperature_model<const InitialTemperature::AsciiPip<dim> >();
           in_OOC = ic.is_inside_2D_polygon(1,position);
         }
    // To compensate for the missing crust in the slab, we change the viscosity law for depths less than 25 km
    // use the def types of the Nubian plate
    if (position.norm() >= 6371000.0-25000.0)
    {
        if (in_OOC)
    	deformation_types[n_fields-2] = deformation_types[n_fields-13]; // slabs last, crust same as OOC/normal crust
        else
    	deformation_types[n_fields-2] = deformation_types[n_fields-8]; // slabs last, crust same as WZ crust
    }
    else
    {
    	//deformation_types[1] = deformation_types[6];
    	deformation_types[n_fields-2] = wet_olivine;
    }

    if (position.norm() >= 6371000.0-25000.0)
    {
        if (in_OOC)
    	deformation_types[n_fields-1] = deformation_types[n_fields-13];
        else
    	deformation_types[n_fields-1] = deformation_types[n_fields-6];
    }
    else
    {
    	//deformation_types[2] = deformation_types[6];
    	deformation_types[n_fields-1] = wet_olivine;
    }

    // Sticky-air
//      if(this->get_geometry_model().depth(position) <= 2000.0)
//    	  deformation_types[0] = constant19;

    // TODO: add get_nonlinear_iteration_number() function
//      if (this->get_nonlinear_iteration_number() == 0 && this->get_timestep_number() == 0)
//    if (strain_rate.norm() == 0 && this->get_timestep_number() == 0)
//    {
//        switch (viscosity_averaging)
//        {
//        case arithmetic:
//        {
//            for (unsigned int i=0; i< volume_fractions.size(); ++i)
//                viscosity += volume_fractions[i]*initial_viscosities[i];
//            break;
//        }
//        case harmonic:
//        {
//            for (unsigned int i=0; i< volume_fractions.size(); ++i)
//                viscosity += volume_fractions[i]/(initial_viscosities[i]);
//            viscosity = 1.0/viscosity;
//            break;
//        }
//        case geometric:
//        {
//            for (unsigned int i=0; i < volume_fractions.size(); ++i)
//                viscosity += volume_fractions[i]*std::log(initial_viscosities[i]);
//            viscosity = std::exp(viscosity);
//            break;
//        }
//        case maximum_composition:
//        {
//            const unsigned int i = (unsigned int)(std::max_element( volume_fractions.begin(),
//                                                  volume_fractions.end() )
//                                                  - volume_fractions.begin());
//            viscosity = initial_viscosities[i];
//            break;
//        }
//        default:
//        {
//            AssertThrow( false, ExcNotImplemented() );
//            break;
//        }
//        }
//    }
//    else
//    {
        // Calculate the second invariant of the deviatoric strain rate tensor
        const SymmetricTensor<2,dim> strain_rate_dev = deviator(strain_rate);
        const double strain_rate_dev_inv = (strain_rate.norm() == 0) ? (position.norm()<5711000. ? 5e-17 : (position.norm()<6271000. ? 1e-15 : 1e-17)) : std::sqrt(0.5) * strain_rate_dev.norm();


        std::vector<double> visc_diffusion_inverse(n_fields);
        std::vector<double> visc_dislocation_inverse(n_fields);
        std::vector<double> visc_Peierls_inverse(n_fields);
        std::vector<double> visc_viscous(n_fields);
        std::vector<double> visc_plastic(n_fields);
        std::vector<double> visc_effective(n_fields);

//
//
//
//
        // Phase transition properties for 660
        const double transition_pressure = 2.30725e10;
        const double transition_temperature = 1855.0;
        const double pressure_width = 1.37340e9; //3500*9.81*40000=rho*g*h
        const double clayperon_slope = -2.5;

        // then calculate the deviation from the transition point (both in temperature
        // and in pressure)
        const double pressure_deviation = pressure - transition_pressure
                                          - clayperon_slope * (temperature - transition_temperature);

        // last, calculate the percentage of material that has undergone the transition
        const double phase_func = 0.5*(1.0 + std::tanh(pressure_deviation / pressure_width));
//          if(phase_func > 0.0 && phase_func < 1.0)
//          std::cout << "phase function " << phase_func << " depth " << 6371000.0 - position.norm() << std::endl;

        // Completely lower mantle, use only diffusion
        if(do_660_visc_smoothing)
            if(phase_func >= 1.0)
            {
                deformation_types[0] = lower_mantle_deformation_type;
//                this->get_pcout() << "> 660 smoothing " << std::endl;
            }

//
        // If not, compute upper mantle viscosity
        for (unsigned int n=0; n<n_fields; n++)
        {
            visc_diffusion_inverse[n] = 1.0 / diffusion(deformation_types[n],
                                        temperature,
                                        pressure);

            visc_dislocation_inverse[n] = 1.0 / dislocation(deformation_types[n],
                                          temperature,
                                          pressure,
                                          strain_rate_dev_inv);

            visc_Peierls_inverse[n] = 1.0 / Peierls(deformation_types[n],
                                                    temperature,
                                                    pressure,
                                                    strain_rate_dev_inv);

            visc_viscous[n] = 1.0 / (visc_diffusion_inverse[n] + visc_dislocation_inverse[n] + visc_Peierls_inverse[n]);

            visc_plastic[n] =  plastic(deformation_types[n],
                                       pressure,
                                       strain_rate_dev_inv);

            visc_effective[n] = 1.0 / ((1.0/visc_plastic[n]) + (1.0/visc_viscous[n]));
//              visc_effective[n] = std::min(visc_viscous[n],visc_plastic[n]);
        }

        // and if necessary add LM viscosity to it based on phase function
        if(do_660_visc_smoothing)
            if (phase_func > 0.0 && phase_func < 1.0)
            {
//            	std::cout << "< 660 smoothing " << std::endl;
                    deformation_types[0] = lower_mantle_deformation_type;

                visc_diffusion_inverse[0] = 1.0 / diffusion(deformation_types[0],
                                            temperature,
                                            pressure);

                visc_dislocation_inverse[0] = 1.0 / dislocation(deformation_types[0],
                                              temperature,
                                              pressure,
                                              strain_rate_dev_inv);

                visc_Peierls_inverse[0] = 1.0 / Peierls(deformation_types[0],
                                                        temperature,
                                                        pressure,
                                                        strain_rate_dev_inv);

                visc_viscous[0] = 1.0 / (visc_diffusion_inverse[0] + visc_dislocation_inverse[0] + visc_Peierls_inverse[0]);

                visc_plastic[0] =  plastic(deformation_types[0],
                                           pressure,
                                           strain_rate_dev_inv);

//              std::cout << "eta eff 0 " << visc_effective[0] << std::endl;
                visc_effective[0] =        phase_func  * (1.0 / ((1.0/visc_plastic[0]) + (1.0/visc_viscous[0]))) +
                                           (1.0 - phase_func) * visc_effective[0];
//              std::cout << "eta eff 1 " << (1.0 / ((1.0/visc_plastic[0]) + (1.0/visc_viscous[0])))  << std::endl;

            }

        //visc_effective[n_fields-2] = std::min(visc_effective[n_fields-2],5e22);
        // Adjust Aegean/Anatolian strength when in thinned region
//        bool in_Anatolia = false;
//        if (InitialTemperature::AsciiPip<dim> *ic = dynamic_cast<InitialTemperature::AsciiPip<dim> *>
//          (const_cast<InitialTemperature::Interface<dim> *>(&this->get_initial_temperature())))
//          {
//            in_Anatolia = ic->is_inside_2D_polygon(0,position);
//          }
//        if (in_Anatolia)
//          {
//           visc_effective[1] *= viscosity_function->value(position,1);
//           visc_effective[2] *= viscosity_function->value(position,2);
//          }

        // Adjust viscosity of other fields locally if wanted
        for (unsigned int v=0; v<n_fields; ++v) 
           visc_effective[v] *= viscosity_function->value(position,v); 

        switch (viscosity_averaging)
        {
        case arithmetic:
        {
            for (unsigned int i=0; i< volume_fractions.size(); ++i)
                viscosity += volume_fractions[i]*visc_effective[i];
            break;
        }
        case harmonic:
        {
            for (unsigned int i=0; i< volume_fractions.size(); ++i)
                viscosity += volume_fractions[i]/(visc_effective[i]);
            viscosity = 1.0/viscosity;
            break;
        }
        case geometric:
        {
            for (unsigned int i=0; i < volume_fractions.size(); ++i)
                viscosity += volume_fractions[i]*std::log(visc_effective[i]);
            viscosity = std::exp(viscosity);
            break;
        }
        case maximum_composition:
        {
            const unsigned int i = (unsigned int)(std::max_element( volume_fractions.begin(),
                                                  volume_fractions.end() )
                                                  - volume_fractions.begin());
            viscosity = visc_effective[i];

            break;
        }
        default:
        {
            AssertThrow( false, ExcNotImplemented() );
            break;
        }
        }
//    }

//    if (viscosity < minimum_visc || viscosity > maximum_visc)
    // Cutoff viscosity between user-specified minimum and maximum
    viscosity = 1.0 / ((1.0 / viscosity) + (1.0 / maximum_visc));
    viscosity = std::max(viscosity, minimum_visc);
//      viscosity = 1.0 / ((1.0 / (viscosity+minimum_visc)) + (1.0 / maximum_visc));
//      viscosity += minimum_visc;
//      viscosity = std::min(std::max(viscosity,minimum_visc),maximum_visc);


    AssertThrow(!isnan(viscosity), ExcMessage("Somehow visc is NaN"));
    return viscosity;
}

template <int dim>
double
MulticomponentVP<dim>::
viscosity_ratio (const double temperature,
                 const double pressure,
                 const std::vector<double> &composition,       /*composition*/
                 const SymmetricTensor<2,dim> &strain_rate,
                 const Point<dim> &position) const
{
    std::vector<double> volume_fractions = compute_volume_fractions(composition);

    // Calculate the second invariant of the deviatoric strain rate tensor
    const SymmetricTensor<2,dim> strain_rate_dev = deviator(strain_rate);
    const double strain_rate_dev_inv = std::sqrt(0.5) * strain_rate_dev.norm();


    // correct for wrong weak zones
    if(do_660_visc_smoothing)
    {
    	deformation_types[0] = upper_mantle_deformation_type;
    }

    // To compensate for the missing crust in the slab, we change the viscosity law for depths less than 25 km
    // use the def types of the Nubian plate
    if (position.norm() >= 6371000.0-25000.0)
    {
    	//deformation_types[1] = deformation_types[5];
    	deformation_types[n_fields-2] = deformation_types[3];
    }
    else
    {
    	//deformation_types[1] = deformation_types[6];
    	deformation_types[n_fields-2] = deformation_types[4];
    }
    if (position.norm() >= 6371000.0-25000.0)
    {
    	//deformation_types[2] = deformation_types[5];
    	deformation_types[n_fields-1] = deformation_types[3];
    }
    else
    {
    	//deformation_types[2] = deformation_types[6];
    	deformation_types[n_fields-1] = deformation_types[4];
    }

    if(position.norm() < 5711000.0)
        deformation_types[0] = lower_mantle_deformation_type;
    else
        deformation_types[0] = upper_mantle_deformation_type;

    std::vector<double> visc_diffusion(n_fields);
    std::vector<double> visc_dislocation(n_fields);
    std::vector<double> visc_Peierls(n_fields);
    std::vector<double> visc_plastic(n_fields);

    for (unsigned int n=0; n<n_fields; n++)
    {
        visc_diffusion[n] = diffusion(deformation_types[n],
                                      temperature,
                                      pressure);

        visc_dislocation[n] = dislocation(deformation_types[n],
                                          temperature,
                                          pressure,
                                          strain_rate_dev_inv);

        visc_Peierls[n] = Peierls(deformation_types[n],
                                  temperature,
                                  pressure,
                                  strain_rate_dev_inv);

        visc_plastic[n] =  plastic(deformation_types[n],
                                   pressure,
                                   strain_rate_dev_inv);
    }



    double viscosity_diff = 0.0, viscosity_disl = 0.0, viscosity_Prls = 0.0, viscosity_plas = 0.0;

    switch (viscosity_averaging)
    {
    case arithmetic:
    {
        for (unsigned int i=0; i< volume_fractions.size(); ++i)
        {
            viscosity_diff += volume_fractions[i]*visc_diffusion[i];
            viscosity_disl += volume_fractions[i]*visc_dislocation[i];
            viscosity_Prls += volume_fractions[i]*visc_Peierls[i];
            viscosity_plas += volume_fractions[i]*visc_plastic[i];
        }
        break;
    }
    case harmonic:
    {
        for (unsigned int i=0; i< volume_fractions.size(); ++i)
        {
            viscosity_diff += volume_fractions[i]/(visc_diffusion[i]);
            viscosity_disl += volume_fractions[i]/(visc_dislocation[i]);
            viscosity_Prls += volume_fractions[i]/(visc_Peierls[i]);
            viscosity_plas += volume_fractions[i]/(visc_plastic[i]);
        }
        viscosity_diff = 1.0/viscosity_diff;
        viscosity_disl = 1.0/viscosity_disl;
        viscosity_Prls = 1.0/viscosity_Prls;
        viscosity_plas = 1.0/viscosity_plas;
        break;
    }
    case geometric:
    {
        for (unsigned int i=0; i < volume_fractions.size(); ++i)
        {
            viscosity_diff += volume_fractions[i]*std::log(visc_diffusion[i]);
            viscosity_disl += volume_fractions[i]*std::log(visc_dislocation[i]);
            viscosity_Prls += volume_fractions[i]*std::log(visc_Peierls[i]);
            viscosity_plas += volume_fractions[i]*std::log(visc_plastic[i]);
        }
        viscosity_diff = std::exp(viscosity_diff);
        viscosity_disl = std::exp(viscosity_disl);
        viscosity_Prls = std::exp(viscosity_Prls);
        viscosity_plas = std::exp(viscosity_plas);
        break;
    }
    case maximum_composition:
    {
        const unsigned int i = (unsigned int)(std::max_element( volume_fractions.begin(),
                                              volume_fractions.end() )
                                              - volume_fractions.begin());
        viscosity_diff = visc_diffusion[i];
        viscosity_disl = visc_dislocation[i];
        viscosity_Prls = visc_Peierls[i];
        viscosity_plas = visc_plastic[i];

        break;
    }
    default:
    {
        AssertThrow( false, ExcNotImplemented() );
        break;
    }
    }

    unsigned int viscosity_ratio = 0.0;
    if (viscosity_plas < viscosity_diff && viscosity_plas < viscosity_disl && viscosity_plas < viscosity_Prls)
        viscosity_ratio = 3;
    if (viscosity_diff < viscosity_plas && viscosity_diff < viscosity_disl && viscosity_diff < viscosity_Prls)
        viscosity_ratio = 0;
    if (viscosity_disl < viscosity_diff && viscosity_disl < viscosity_plas && viscosity_disl < viscosity_Prls)
        viscosity_ratio = 1;
    if (viscosity_Prls < viscosity_diff && viscosity_Prls < viscosity_plas && viscosity_Prls < viscosity_disl)
        viscosity_ratio = 2;


    return viscosity_ratio;
}

template <int dim>
double
MulticomponentVP<dim>::
reference_viscosity () const
{
    //TODO: change?
    return initial_viscosities[0]; //background
}

template <int dim>
double
MulticomponentVP<dim>::
reference_density () const
{
    return upper_mantle.rho_0;  //background
}

template <int dim>
double
MulticomponentVP<dim>::
reference_thermal_expansion_coefficient () const
{
    return upper_mantle.alpha; //background
}

template <int dim>
double
MulticomponentVP<dim>::
specific_heat (const double,
               const double,
               const std::vector<double> &composition,
               const Point<dim> &) const
{
    std::vector<double> volume_fractions = compute_volume_fractions(composition);

    double cp = 0.0;
    cp += volume_fractions[0]*material_types[0].c_P;


    //Arithmetic averaging of specific heats
    for (unsigned int i=1; i< volume_fractions.size(); ++i)
        cp += volume_fractions[i]*material_types[i].c_P;

    return cp;
}

template <int dim>
double
MulticomponentVP<dim>::
reference_cp () const
{
    return upper_mantle.c_P; //background
}

template <int dim>
double
MulticomponentVP<dim>::
thermal_conductivity (const double,
                      const double,
                      const std::vector<double> &composition,
                      const Point<dim> &) const
{
    std::vector<double> volume_fractions = compute_volume_fractions(composition);
    double k = 0.0;
    k += volume_fractions[0]*material_types[0].k;



    //Arithmetic averaging of thermal conductivities
    //This may not be strictly the most reasonable thing,
    //but for most Earth materials we hope that they do
    //not vary so much that it is a big problem.
    for (unsigned int i=1; i< volume_fractions.size(); ++i)
        k += volume_fractions[i]*material_types[i].k;

    return k;
}

template <int dim>
double
MulticomponentVP<dim>::
reference_thermal_diffusivity () const
{
    return upper_mantle.k /( upper_mantle.rho_0* upper_mantle.c_P ); //background
}

template <int dim>
double
MulticomponentVP<dim>::
density (const double temperature,
         const double,
         const std::vector<double> &composition,
         const Point<dim> &) const
{
    std::vector<double> volume_fractions = compute_volume_fractions(composition);
    double rho = 0.0;
    const double temperature_factor = 1.0 - (material_types[0].alpha * (temperature - material_types[0].T_0));
    rho += volume_fractions[0] * material_types[0].rho_0 * temperature_factor;



    //Arithmetic averaging of densities
    for (unsigned int i=1; i< volume_fractions.size(); ++i)
    {
        //not strictly correct if thermal expansivities are different, since we are interpreting
        //these compositions as volume fractions, but the error introduced should not be too bad.
        const double temperature_factor = 1.0 - (material_types[i].alpha * (temperature - material_types[i].T_0));
        rho += volume_fractions[i] * material_types[i].rho_0 * temperature_factor;
    }

    return rho;
}


template <int dim>
double
MulticomponentVP<dim>::
thermal_expansion_coefficient (const double,
                               const double,
                               const std::vector<double> &composition,
                               const Point<dim> &) const
{
    std::vector<double> volume_fractions = compute_volume_fractions(composition);
    double alpha = 0.0;
    alpha += volume_fractions[0]*material_types[0].alpha;


    //Arithmetic averaging of thermal expansivities
    for (unsigned int i=1; i< volume_fractions.size(); ++i)
        alpha += volume_fractions[i]*material_types[i].alpha;

    return alpha;
}


template <int dim>
double
MulticomponentVP<dim>::
compressibility (const double,
                 const double,
                 const std::vector<double> &, /*composition*/
                 const Point<dim> &) const
{
    return 0.0; //incompressible
}

template <int dim>
bool
MulticomponentVP<dim>::
viscosity_depends_on (const NonlinearDependence::Dependence dependence) const
{
    if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none))
        return true;
    else if (((dependence & NonlinearDependence::pressure) != NonlinearDependence::none))
        return true;
    else if (((dependence & NonlinearDependence::temperature) != NonlinearDependence::none))
        return true;
    else if (((dependence & NonlinearDependence::strain_rate) != NonlinearDependence::none))
        return true;
    else
        return false;
}


template <int dim>
bool
MulticomponentVP<dim>::
density_depends_on (const NonlinearDependence::Dependence dependence) const
{
    if (((dependence & NonlinearDependence::temperature) != NonlinearDependence::none))
        return true;
    else if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none))
        return true;
    else
        return false;
}

template <int dim>
bool
MulticomponentVP<dim>::
compressibility_depends_on (const NonlinearDependence::Dependence) const
{
    return false;
}

template <int dim>
bool
MulticomponentVP<dim>::
specific_heat_depends_on (const NonlinearDependence::Dependence dependence) const
{
    if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none))
        return true;
    else
        return false;
}

template <int dim>
bool
MulticomponentVP<dim>::
thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const
{
    if (((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none))
        return true;
    else
        return false;
}


template <int dim>
bool
MulticomponentVP<dim>::
is_compressible () const
{
    return false;
}


template <int dim>
void
MulticomponentVP<dim>::declare_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Material model");
    {
        prm.enter_subsection("Multicomponent");
        {
            prm.declare_entry ("Reference temperature", "293",
                               Patterns::Double (0),
                               "The reference temperature $T_0$. Units: $K$.");
            prm.declare_entry ("Densities", "3300.",
                               Patterns::List(Patterns::Double(0)),
                               "List of densities for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value.  Units: $kg / m^3$");
            prm.declare_entry ("Initial viscosities", "1.e21",
                               Patterns::List(Patterns::Double(0)),
                               "List of initial (first NI) viscosities for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $Pa s$");
            prm.declare_entry ("Thermal expansivities", "4.e-5",
                               Patterns::List(Patterns::Double(0)),
                               "List of thermal expansivities for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $1/K$");
            prm.declare_entry ("Specific heats", "1250.",
                               Patterns::List(Patterns::Double(0)),
                               "List of specific heats for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $J /kg /K$");
            prm.declare_entry ("Thermal conductivities", "4.7",
                               Patterns::List(Patterns::Double(0)),
                               "List of thermal conductivities for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Prefactors diffusion", "",
                               Patterns::List(Patterns::Double(0)),
                               "List of prefactors of diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Activation energies diffusion", "",
                               Patterns::List(Patterns::Double(0)),
                               "List of activation energies diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Activation volumes diffusion", "4.7",
                               Patterns::List(Patterns::Double(0)),
                               "List of activation volumes diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Nus diffusion", "1",
                               Patterns::List(Patterns::Double(0)),
                               "List of constant factors diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Prefactors dislocation", "",
                               Patterns::List(Patterns::Double(0)),
                               "List of prefactors of dislocation for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Activation energies dislocation", "",
                               Patterns::List(Patterns::Double(0)),
                               "List of activation energies dislocation for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Activation volumes dislocation", "4.7",
                               Patterns::List(Patterns::Double(0)),
                               "List of activation volumes diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Nus dislocation", "1",
                               Patterns::List(Patterns::Double(0)),
                               "List of constant factors diffusion for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Stress exponents", "3",
                               Patterns::List(Patterns::Double(0)),
                               "List of stress exponents dislocation for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Friction angles", "10",
                               Patterns::List(Patterns::Double(0)),
                               "List of friction angles plasticity for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Cohesions", "1e9",
                               Patterns::List(Patterns::Double(0)),
                               "List of cohesions plasticity for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $W/m/K$ ");
            prm.declare_entry ("Minimum viscosity", "1e20",
                               Patterns::Double (0),
                               "The value of the minimum viscosity cutoff. Units: $kg/m/s$.");
            prm.declare_entry ("Maximum viscosity", "1e25",
                               Patterns::Double (0),
                               "The value of the maximum viscosity cutoff. Units: $kg/m/s$.");
            prm.declare_entry("Viscosity averaging scheme", "harmonic",
                              Patterns::Selection("arithmetic|harmonic|geometric|maximum composition"),
                              "When more than one compositional field is present at a point "
                              "with different viscosities, we need to come up with an average "
                              "viscosity at that point.  Select a weighted harmonic, arithmetic, "
                              "geometric, or maximum composition.");

            prm.declare_entry ("Use 660 viscosity smoothing", "false",
                               Patterns::Bool (),
                               "Whether or not to use a smooth transition from diffusion-only "
                               "lower mantle viscosity to composite upper mantle viscosity.");
            prm.declare_entry ("Use family B of C12", "false",
                               Patterns::Bool (),
                               "Whether or not to use family B diffusion parameters "
                               "instead of A to compute lower mantle viscosity.");
            prm.declare_entry ("List of deformation types", "",
                               Patterns::List (Patterns::Selection("All wet olivine|Wet olivine|Dry olivine|Wet quartz|Plagioclase|Diffusion|Constant19|Constant20|Constant21|Constant22|Constant24|Fault|Wedge")),
                               "A list of deformation types for each composition.");
            prm.declare_entry ("Lower mantle deformation type", "Diffusion",
                               Patterns::List (Patterns::Selection("All wet olivine|Wet olivine|Dry olivine|Wet quartz|Plagioclase|Diffusion|Constant19|Constant20|Constant21|Constant22|Constant24|Fault|Wedge")),
                               "The deformation type of the lower mantle.");
            prm.declare_entry ("Diffusion scale factor beta","4",
                               Patterns::Double (1),
                               "The value of the scale factor beta in the diffusion law." );
            prm.declare_entry ("Dislocation scale factor beta","4",
                               Patterns::Double (1),
                               "The value of the scale factor beta in the dislocation law." );


            prm.declare_entry ("List of material types", "",
                               Patterns::List (Patterns::Selection("Oceanic crust|Continental crust|Upper mantle|Transition mantle|Lower mantle|Air|Ocean")),
                               "A list of material types for each composition.");
            Functions::ParsedFunction<dim>::declare_parameters(prm,1);
            prm.declare_entry("Function expression","1.0");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}



template <int dim>
void
MulticomponentVP<dim>::parse_parameters (ParameterHandler &prm)
{
    this->get_pcout() << "Really parsing viscosity parameters " << std::endl;
    //not pretty, but we need to get the number of compositional fields before
    //simulatoraccess has been initialized here...
    prm.enter_subsection ("Compositional fields");
    {
        n_fields = prm.get_integer ("Number of fields");
        n_compositional_fields = n_fields;
    }
    prm.leave_subsection();
    n_fields++; //increment for background

    prm.enter_subsection("Material model");
    {
        prm.enter_subsection("Multicomponent");
        {
            reference_T                = prm.get_double ("Reference temperature");

            if (prm.get ("Viscosity averaging scheme") == "harmonic")
                viscosity_averaging = harmonic;
            else if (prm.get ("Viscosity averaging scheme") == "arithmetic")
                viscosity_averaging = arithmetic;
            else if (prm.get ("Viscosity averaging scheme") == "geometric")
                viscosity_averaging = geometric;
            else if (prm.get ("Viscosity averaging scheme") == "maximum composition")
                viscosity_averaging = maximum_composition;

            minimum_visc       = prm.get_double ("Minimum viscosity");
            maximum_visc       = prm.get_double ("Maximum viscosity");

            std::vector<double> x_values;

            //Parse initial viscosities
            x_values = Utilities::string_to_double(Utilities::split_string_list(prm.get ("Initial viscosities")));
            AssertThrow(x_values.size() == 1u || (x_values.size() == n_fields),
                        ExcMessage("Length of initial viscosity list must be either one, or n_compositional_fields+1"));
            if (x_values.size() == 1)
                initial_viscosities.assign( n_fields , x_values[0]);
            else
                initial_viscosities = x_values;

            do_660_visc_smoothing = prm.get_bool("Use 660 viscosity smoothing");
            family_b = prm.get_bool("Use family B of C12");

            try
            {
               //viscosity_function.parse_parameters(prm);
               viscosity_function.reset (new Functions::ParsedFunction<dim>(n_fields));
               viscosity_function->parse_parameters(prm);
            }
            catch (...)
            {
               std::cerr << "FunctionParser failed to parse\n"
                         << "\t Viscosity weak zone function\n"
                         << "with expression \n"
                          << "\t' " << prm.get("Function expression") << "'";
               throw;
            }

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    // Setting the material deformation parameters using KW93, R95, TS02
//      all_wet_olivine.n = wet_olivine.n = 3.0;
//      all_wet_olivine.A_diff = wet_olivine.A_diff = 3.73e-14;
//     all_wet_olivine.A_disl = wet_olivine.A_disl = 3.906e-15;
//      all_wet_olivine.Q_diff = wet_olivine.Q_diff = 24e4;
//      all_wet_olivine.Q_disl = wet_olivine.Q_disl = 43e4;
//      all_wet_olivine.V_diff = wet_olivine.V_diff = 5e-6;
//      all_wet_olivine.V_disl = wet_olivine.V_disl = 15e-6;
//      all_wet_olivine.C = wet_olivine.C = 1e6;
//      all_wet_olivine.phi = wet_olivine.phi = 20.0*numbers::PI/180.0;
//      all_wet_olivine.A_Peierls = 1e-150;
    all_wet_olivine.n = wet_olivine.n = 3.5;
    all_wet_olivine.A_diff = wet_olivine.A_diff = 1e-9;
    all_wet_olivine.A_disl = wet_olivine.A_disl = 3.1e-17;
    all_wet_olivine.Q_diff = wet_olivine.Q_diff = 3.35e5;
    all_wet_olivine.Q_disl = wet_olivine.Q_disl = 4.8e5;
    all_wet_olivine.V_diff = wet_olivine.V_diff = 4.8e-6;
    all_wet_olivine.V_disl = wet_olivine.V_disl = 11e-6;
    all_wet_olivine.C = wet_olivine.C = 1e6;
    all_wet_olivine.phi = wet_olivine.phi = 20.0*numbers::PI/180.0;
    all_wet_olivine.A_Peierls = 1e-150;
    //all_wet_olivine.nu_diff = wet_olivine.nu_diff = 6.0;
    //all_wet_olivine.nu_disl = wet_olivine.nu_disl = 6.0;
    all_wet_olivine.nu_diff = wet_olivine.nu_diff = 4.0;
    all_wet_olivine.nu_disl = wet_olivine.nu_disl = 4.0;

    dry_olivine.n = 3.5;
    dry_olivine.A_diff = 6.079e-14;
    dry_olivine.A_disl = 2.42e-16;
    dry_olivine.Q_diff = 3e5;
    dry_olivine.Q_disl = 54e4;
    dry_olivine.V_diff = 6e-6;
    dry_olivine.V_disl = 20e-6;
    dry_olivine.C = 1e6;
    dry_olivine.phi = 20.0*numbers::PI/180.0;

    // M13
    wet_quartz.n = 2.3;
    wet_quartz.A_diff = 1e-40;
    wet_quartz.A_disl = 1e-3;
    wet_quartz.Q_diff = 0.0;
    wet_quartz.Q_disl = 154e3;
    wet_quartz.V_diff = 0.0;
    wet_quartz.V_disl = 0.0;
    wet_quartz.C = 1e6;
    wet_quartz.phi = 20.0*numbers::PI/180.0;

    // Ranalli 2000
    plag_anor_75.n = 3.2;
    plag_anor_75.A_diff = 1e-40;
    plag_anor_75.A_disl = 2.02e-23;
    plag_anor_75.Q_diff = 0.0;
    plag_anor_75.Q_disl = 238e3;
    plag_anor_75.V_diff = 0.0;
    plag_anor_75.V_disl = 0.0;
    plag_anor_75.C = 1e7;
    plag_anor_75.phi = 20.0*numbers::PI/180.0;

    // Cizkova 2012 Family A
    diffusion_only.n = 1.0;
    //diffusion_only.A_diff = 5e-17;
    //diffusion_only.A_disl = 5e-17;
    diffusion_only.A_diff = 7e-17;
    diffusion_only.A_disl = 7e-17;
    diffusion_only.Q_diff = 20e4;
    diffusion_only.Q_disl = 20e4;
    diffusion_only.V_diff = 1.1e-6;
    diffusion_only.V_disl = 1.1e-6;
    diffusion_only.nu_diff = 4.0;
    diffusion_only.nu_disl = 4.0;
    diffusion_only.C = 1e15;
    diffusion_only.phi = 0.0;

    // Cizkova 2012 Family B
    if (family_b)
    {
    diffusion_only.n = 1.0;
    diffusion_only.A_diff = 1.2e-14;
    diffusion_only.A_disl = 1.2e-14;
    diffusion_only.Q_diff = 20e4;
    diffusion_only.Q_disl = 20e4;
    diffusion_only.V_diff = 2.2e-6;
    diffusion_only.V_disl = 2.2e-6;
    diffusion_only.nu_diff = 4.0;
    diffusion_only.nu_disl = 4.0;
    diffusion_only.C = 1e15;
    diffusion_only.phi = 0.0;
    }

    // Newtonian viscosities
    constant19.n      = constant20.n = constant21.n = constant22.n = constant24.n = 1.0;
    constant19.A_disl = constant19.A_diff = 1e-19;
    constant20.A_disl = constant20.A_diff = 1e-20;
    constant21.A_disl = constant21.A_diff = 1e-21;
    constant22.A_disl = constant22.A_diff = 1e-22;
    constant24.A_disl = constant24.A_diff = 1e-24;
    constant19.Q_diff = constant20.Q_diff = constant21.Q_diff = constant22.Q_diff = constant24.Q_diff = 0.0;
    constant19.Q_disl = constant20.Q_disl = constant21.Q_disl = constant22.Q_disl = constant24.Q_disl = 0.0;
    constant19.V_diff = constant20.V_diff = constant21.V_diff = constant22.V_diff = constant24.V_diff = 0.0;
    constant19.V_disl = constant20.V_disl = constant21.V_disl = constant22.V_disl = constant24.V_disl = 0.0;
    constant19.nu_diff = constant20.nu_diff = constant21.nu_diff = constant22.nu_diff = constant24.nu_diff = 4.0;
    constant19.nu_disl = constant20.nu_disl = constant21.nu_disl = constant22.nu_disl = constant24.nu_disl = 4.0;
    constant19.C      = constant20.C = constant21.C = constant22.C = constant24.C = 1e15;
    constant19.phi    = constant20.phi = constant21.phi = constant22.phi = constant24.phi = 0.0;

    // Weak zone
    mohr_coulomb_weak_zone.n      = mohr_coulomb_wedge.n      = 3.5;
    mohr_coulomb_weak_zone.A_diff = mohr_coulomb_wedge.A_diff = 1e-9;
    mohr_coulomb_weak_zone.A_disl = mohr_coulomb_wedge.A_disl = 3.1e-17;
    mohr_coulomb_weak_zone.Q_diff = mohr_coulomb_wedge.Q_diff = 3.35e5;
    mohr_coulomb_weak_zone.Q_disl = mohr_coulomb_wedge.Q_disl = 4.8e5;
    mohr_coulomb_weak_zone.V_diff = mohr_coulomb_wedge.V_diff = 4.8e-6;
    mohr_coulomb_weak_zone.V_disl = mohr_coulomb_wedge.V_disl = 11e-6;
    mohr_coulomb_weak_zone.nu_diff = mohr_coulomb_wedge.nu_diff = 2.0;
    mohr_coulomb_weak_zone.nu_disl = mohr_coulomb_wedge.nu_disl = 2.0;
    mohr_coulomb_weak_zone.C      = 5e5;
    mohr_coulomb_wedge.C      = 5e6;
    mohr_coulomb_weak_zone.phi    = 0.0*numbers::PI/180.0;
    mohr_coulomb_wedge.phi    = 2.0*numbers::PI/180.0;




    // Setting the material parameters using the Schubert, Turcotte and Olson book
    // Table 3.1
    sticky_air.rho_0 = 1.0;
    sticky_air.k = 200.0;
    sticky_air.c_P = 1e-3;
    ocean.rho_0 = 1030.0; //Wikipedia, maybe ask Dirk
    ocean.k = 200.0;
    ocean.c_P = 1e-3;
    oceanic_crust.rho_0 = 3000.0;
    continental_crust.rho_0 = 2700.0;
    upper_mantle.rho_0 = 3350.0;
    transition_mantle.rho_0 = 3860.0;
    lower_mantle.rho_0 = 4870.0;

    prm.enter_subsection("Material model");
    {
        prm.enter_subsection("Multicomponent");
        {
            constant20.nu_diff = prm.get_double("Diffusion scale factor beta");
            constant20.nu_disl = prm.get_double("Dislocation scale factor beta");
            // The type of material and deformation for the mantle and each field
            std::vector<std::string> m_types = dealii::Utilities::split_string_list(prm.get("List of material types"));
            AssertThrow(m_types.size() == 1u || m_types.size() == n_fields,
                        ExcMessage(std::string("Length of material type list should equal 1 or the number of fields +1 instead of ")
                                   + dealii::Utilities::int_to_string(m_types.size())));

            std::vector<std::string> d_types = dealii::Utilities::split_string_list(prm.get("List of deformation types"));
            this->get_pcout() << d_types.size() << std::endl;
            AssertThrow(d_types.size() == 1u || d_types.size() == n_fields,
                        ExcMessage(std::string("Length of deformation type list should equal 1 or the number of fields +1 instead of ")
                                   + dealii::Utilities::int_to_string(d_types.size())));

            this->get_pcout() << "Number of deformation types specified: " << d_types.size() << std::endl;

            std::string lm_type = prm.get("Lower mantle deformation type");

            for (unsigned int i = 0; i < n_fields; i++)
            {
                if (m_types[i] == "Upper mantle")
                {
                    material_types.push_back(upper_mantle);
                }
                else if (m_types[i] == "Air")
                {
                    material_types.push_back(sticky_air);
                }
                else if (m_types[i] == "Oceanic crust")
                {
                    material_types.push_back(oceanic_crust);
                }
                else if (m_types[i] == "Continental crust")
                {
                    material_types.push_back(continental_crust);
                }
                else if (m_types[i] == "Transition mantle")
                {
                    material_types.push_back(transition_mantle);
                }
                else if (m_types[i] == "Lower mantle")
                {
                    material_types.push_back(transition_mantle);
                }
                else if (m_types[i] == "Ocean")
                {
                    material_types.push_back(ocean);
                }
                else
                {
                    AssertThrow (false, ExcNotImplemented());
                }
            }

            for (unsigned int i = 0; i < n_fields; i++)
            {
                if (d_types[i] == "Wet olivine")
                {
                    deformation_types.push_back(wet_olivine);
                }
                else if (d_types[i] == "All wet olivine")
                {
                    deformation_types.push_back(all_wet_olivine);
                }
                else if (d_types[i] == "Dry olivine")
                {
                    deformation_types.push_back(dry_olivine);
                }
                else if (d_types[i] == "Wet quartz")
                {
                    deformation_types.push_back(wet_quartz);
                }
                else if (d_types[i] == "Plagioclase")
                {
                    deformation_types.push_back(plag_anor_75);
                }
                else if (d_types[i] == "Diffusion")
                {
                    deformation_types.push_back(diffusion_only);
                }
                else if (d_types[i] == "Constant19")
                {
                    deformation_types.push_back(constant19);
                }
                else if (d_types[i] == "Constant20")
                {
                    deformation_types.push_back(constant20);
                }
                else if (d_types[i] == "Constant21")
                {
                    deformation_types.push_back(constant21);
                }
                else if (d_types[i] == "Constant22")
                {
                    deformation_types.push_back(constant22);
                }
                else if (d_types[i] == "Constant24")
                {
                    deformation_types.push_back(constant24);
                }
                else if (d_types[i] == "Fault")
                {
                    deformation_types.push_back(mohr_coulomb_weak_zone);
                }
                else if (d_types[i] == "Wedge")
                {
                    deformation_types.push_back(mohr_coulomb_wedge);
                }
                else
                {
                    AssertThrow (false, ExcNotImplemented());
                }
            }
                if (lm_type == "Wet olivine")
                {
                    lower_mantle_deformation_type = wet_olivine;
                }
                else if (lm_type == "All wet olivine")
                {
                    lower_mantle_deformation_type = all_wet_olivine;
                }
                else if (lm_type == "Dry olivine")
                {
                    lower_mantle_deformation_type = dry_olivine;
                }
                else if (lm_type == "Wet quartz")
                {
                    lower_mantle_deformation_type = wet_quartz;
                }
                else if (lm_type == "Plagioclase")
                {
                    lower_mantle_deformation_type = plag_anor_75;
                }
                else if (lm_type == "Diffusion")
                {
                    lower_mantle_deformation_type = diffusion_only;
                }
                else if (lm_type == "Constant19")
                {
                    lower_mantle_deformation_type = constant19;
                }
                else if (lm_type == "Constant20")
                {
                    lower_mantle_deformation_type = constant20;
                }
                else if (lm_type == "Constant21")
                {
                    lower_mantle_deformation_type = constant21;
                }
                else if (lm_type == "Constant22")
                {
                    lower_mantle_deformation_type = constant22;
                }
                else if (lm_type == "Constant24")
                {
                    lower_mantle_deformation_type = constant24;
                }
                else if (lm_type == "Fault")
                {
                    lower_mantle_deformation_type = mohr_coulomb_weak_zone;
                }
                else if (lm_type == "Wedge")
                {
                    lower_mantle_deformation_type = mohr_coulomb_wedge;
                }
                else
                {
                    AssertThrow (false, ExcNotImplemented());
                }

                upper_mantle_deformation_type = deformation_types[0];
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    this->get_pcout() << "Really done parsing viscosity parameters " << std::endl;


}
}
}

// explicit instantiations
namespace aspect
{
namespace MaterialModel
{
ASPECT_REGISTER_MATERIAL_MODEL(MulticomponentVP,
                               "multicomponent vp",
                               "This model is for use with an arbitrary number of compositional fields, where each field"
                               " represents a rock type which can have completely different properties from the others."
                               " However, each rock type itself has constant material properties.  The value of the "
                               " compositional field is interpreted as a volume fraction. If the sum of the fields is"
                               " greater than one, they are renormalized.  If it is less than one, material properties "
                               " for ``background mantle'' make up the rest. When more than one field is present, the"
                               " material properties are averaged arithmetically.  An exception is the viscosity,"
                               " where the averaging should make more of a difference.  For this, the user selects"
                               " between arithmetic, harmonic, geometric, or maximum composition averaging.")
}
}

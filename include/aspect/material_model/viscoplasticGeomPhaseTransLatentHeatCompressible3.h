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
/*  $Id: Stokes.h 1433 2012-12-08 08:24:55Z bangerth $  */


#ifndef __aspect__model_viscoplasticGeomPhaseTransLatentHeatCompressible3_h
#define __aspect__model_viscoplasticGeomPhaseTransLatentHeatCompressible3_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * A material model that consists of globally constant values for all
     * material parameters except that the density decays linearly with the
     * temperature.
     *
     * The model is considered incompressible, following the definition
     * described in Interface::is_compressible. This is essentially
     * the material model used in the step-32 tutorial program.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class ViscoplasticGeomPhaseTransLatentHeatCompressible3 : public MaterialModel::InterfaceCompatibility<dim>, public ::aspect::SimulatorAccess<dim>
      //    {
      //    {
    {
      public:
////////////////////////////////////////////////////////////////////////////////////////////////////
	double entropy_derivative (const double temperature,
                        	   const double pressure,
                        	   const std::vector<double> &compositional_fields,
                        	   const Point<dim> &position,
                        	   const NonlinearDependence::Dependence dependence) const;

        double reaction_term (const double temperature,
			      const double pressure,
                              const std::vector<double> &compositional_fields,
                              const Point<dim> &position,
                              const unsigned int compositional_variable) const;
////////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * @name Physical parameters used in the basic equations
         * @{
         */
        virtual double viscosity (const double                  temperature,
                                  const double                  pressure,
                                  const std::vector<double>    &compositional_fields,
                                  const SymmetricTensor<2,dim> &strain_rate,
                                  const Point<dim>             &position) const;

        virtual double viscosity_ratio (const double temperature,
                                        const double pressure,
                                        const std::vector<double>    &compositional_fields,
                                        const SymmetricTensor<2,dim> &strain_rate,
                                        const Point<dim> &position) const;

        virtual double density (const double temperature,
                                const double pressure,
                                const std::vector<double> &compositional_fields,
                                const Point<dim> &position) const;

        virtual double compressibility (const double temperature,
                                        const double pressure,
                                        const std::vector<double> &compositional_fields,
                                        const Point<dim> &position) const;

        virtual double specific_heat (const double temperature,
                                      const double pressure,
                                      const std::vector<double> &compositional_fields,
                                      const Point<dim> &position) const;

        virtual double thermal_expansion_coefficient (const double      temperature,
                                                      const double      pressure,
                                                      const std::vector<double> &compositional_fields,
                                                      const Point<dim> &position) const;

        virtual double thermal_conductivity (const double temperature,
                                             const double pressure,
                                             const std::vector<double> &compositional_fields,
                                             const Point<dim> &position) const;
        /**
         * @}
         */

        /**
         * @name Qualitative properties one can ask a material model
         * @{
         */

        /**
        * Return true if the viscosity() function returns something that
        * may depend on the variable identifies by the argument.
        */
        virtual bool
        viscosity_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
        * Return true if the density() function returns something that
        * may depend on the variable identifies by the argument.
        */
        virtual bool
        density_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
        * Return true if the compressibility() function returns something that
        * may depend on the variable identifies by the argument.
        *
        * This function must return false for all possible arguments if the
        * is_compressible() function returns false.
        */
        virtual bool
        compressibility_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
        * Return true if the specific_heat() function returns something that
        * may depend on the variable identifies by the argument.
        */
        virtual bool
        specific_heat_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
        * Return true if the thermal_conductivity() function returns something that
        * may depend on the variable identifies by the argument.
        */
        virtual bool
        thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return whether the model is compressible or not.  Incompressibility
         * does not necessarily imply that the density is constant; rather, it
         * may still depend on temperature or pressure. In the current
         * context, compressibility means whether we should solve the contuity
         * equation as $\nabla \cdot (\rho \mathbf u)=0$ (compressible Stokes)
         * or as $\nabla \cdot \mathbf{u}=0$ (incompressible Stokes).
         */
        virtual bool is_compressible () const;
        /**
         * @}
         */

        /**
         * @name Reference quantities
         * @{
         */
        virtual double reference_viscosity () const;

        virtual double reference_density () const;

        virtual double reference_thermal_expansion_coefficient () const;

//TODO: should we make this a virtual function as well? where is it used?
        double reference_thermal_diffusivity () const;

        double reference_cp () const;
        /**
         * @}
         */

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter
         * file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);
        /**
         * @}
         */


      private:

////////////////////////////////////////////////////////////////////////////////////////////////////
        double phase_transition_function (const double temperature,
                                          const int j,
                                          const Point<dim> &position) const;

	double phase_function_derivative (const double temperature,
                	                  const int j,
                           	          const double pressure,
                               		  const Point<dim> &position) const;
////////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * Function to calculate the compositional field with the highest value for infinite norm averaging.
         */
        int maximum_composition(const std::vector<double> &comp) const;

        /**
         * Function to calculate the harmonic average.
         */
        double harmonic_average(const std::vector<double> &comp,
                                const std::vector<double> &eta) const;

        /**
         * Function to calculate the arithmetic average.
         */
        double arithmetic_average(const std::vector<double> &comp,
                                  const std::vector<double> &eta) const;

        /**
         * Function to calculate the geometric average.
         */
        double geometric_average(const std::vector<double> &comp,
                                 const std::vector<double> &eta) const;

        /*
         * Function to calculate diffusion creep.
         */
        double diffusion(const double prefactor,
                         const double activation_energy,
                         const double activation_volume,
                         const double temperature,
                         const double pressure,
                         const double nu) const;
        /*
         * Function to calculate dislocation creep.
         */
        double dislocation(const double prefactor,
                           const double stress_exponent,
                           const double activation_energy,
                           const double activation_volume,
                           const double temperature,
                           const double pressure,
                           const double strain_rate,
                           const double nu) const;
        /*
         * Function to calculate plasticity.
         */
        double plastic(const double phi,
                       const double cohesion,
                       const double pressure,
                       const double strain_rate) const;

        /*
         * Parsed Function that specifies where to apply additional weakness.
         */
        Functions::ParsedFunction<dim> weak_zone_function;

////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<double>            transition_depths;
        std::vector<double>            Clapeyron_slopes;
        std::vector<double>            transition_widths;
        std::vector<double>            transition_pressures;
////////////////////////////////////////////////////////////////////////////////////////////////////

        /*
         * The number of compositional fields.
         */
        unsigned int                   n_compositional_fields;

        /*
         * List of angles of internal friction of the compositional fields.
         */
        std::vector<double>            phis_fields;

        /*
         * List of thermal conductivities of the compositional fields.
         */
        std::vector<double>            conductivities_fields;

        /*
         * List of heat capacities of the compositional fields.
         */
        std::vector<double>            capacities_fields;

        /*
         * List of reference densities of the compositional fields used in the density calculation.
         */
        std::vector<double>            refdens_fields;

        /*
         * List of reference temperatures of the compositional fields used in the density calculation.
         */
        std::vector<double>            reftemps_fields;

        /*
         * List of cohensions of the compositional fields used in the plasticity calculation.
         */
        std::vector<double>            cohesions_fields;

        /*
         * List of strain rate exponents of the compositional fields used in the calculation of dislocation creep.
         */
        std::vector<double>            stress_exponents_fields;

        /*
         * List of activation volumes of the compositional fields used in the calculation of diffusion creep.
         */
        std::vector<double>            activation_volumes_diffusion_fields;

        /*
         * List of activation energies of the compositional fields used in the calculation of diffusion creep.
         */
        std::vector<double>            activation_energies_diffusion_fields;

        /*
         * List of prefactors of the compositional fields used in the calculation of diffusion creep.
         * This includes water fugacity, water fugaticty exponent, grain size and grain size exponent.
         */
        std::vector<double>            prefactors_diffusion_fields;

        /*
         * List of activation volumes of the compositional fields used in the calculation of dislocation creep.
         */
        std::vector<double>            activation_volumes_dislocation_fields;

        /*
         * List of activation energies of the compositional fields used in the calculation of dislocation creep.
         */
        std::vector<double>            activation_energies_dislocation_fields;

        /*
         * List of prefactors of the compositional fields used in the calculation of dislocation creep.
         * This includes water fugacity, water fugacity exponent, grain size and grain size exponent.
         */
        std::vector<double>            prefactors_dislocation_fields;

        /*
         * A scaling factor for dislocation creep viscosity.
         */
        std::vector<double>            nu_dislocation_fields;

        /*
         * A scaling factor for diffusion creep viscosity.
         */
        std::vector<double>            nu_diffusion_fields;

        /*
         * List of initial viscosities of the compositional fields that are prescribed in the very first nonlinear iteration.
         */
        std::vector<double>            init_eta_fields;

        /**
         * The viscosity used for scaling the governing equations
         */
        double reference_eta;

        /**
         * The viscosity prescribed during the very first nonlinear iteration in absence of compositional fields.
         */
        double initial_eta;

        /**
         * The minimum value of the viscosity cut-off used to restrict the overall viscosity ratio.
         */
        double minimum_eta;

        /**
         * The maximum value of the viscosity cut-off used to restrict the overall viscosity ratio.
         */
        double maximum_eta;

        /**
         * The type of averaging used for the contributions of the compositional fields to viscosity (and density).
         */
        std::string viscosity_averaging;

        /**
         * Whether or not to use strain rate weakening in the calculation of plastic viscosity.
         */
        bool strain_rate_weakening;

        /**
         * Whether or not to use an harmonic average of the viscous and plastic viscosity
         * or to take the minimum.
         */
        bool harmonic_plastic_viscous;

        /**
         * Whether or not to use an harmonic average of the effective and maximum viscosity
         * or to take the minimum. If true, the minimum viscosity is added to the viscosity.
         */
        bool harmonic_max;


        /**
         * The reference strain rate used for strain rate weakening.
         */
        double ref_strain_rate;

        /**
         * Whether or not the apply a function-specified weak zone as a last step in the viscosity calculation.
         */
        bool   weak_zone;

        /**
         * The constant thermal expansivity
         */
        double thermal_alpha;

        /**
         * The thermal conductivity.
         */
        double k_value;

        /**
         * The reference density used for scaling the governing equations and in absence of compositional fields.
         */
        double reference_rho;

        /**
         * The reference specific heat used in calculating the reference thermal diffusivity and in absence of compositional fields.
         */
        double reference_specific_heat;

        /**
         * The reference temperature used in the density calculation in absence of compositional fields.
         */
        double reference_T;

        /**
         * The activation energy used in the calculation of diffusion creep.
         */
        double activation_energy_diffusion;

        /**
         * The activation volume used in the calculation of diffusion creep.
         */
        double activation_volume_diffusion;

        /**
         * The prefactor used in the calculation of diffusion creep, including water fugacity,
         * water fugacity exponent, grain size and grain size exponent.
         */
        double prefactor_diffusion;
        /**
         * The activation energy used in the calculation of dislocation creep.
         */
        double activation_energy_dislocation;

        /**
         * The activation volume used in the calculation of diffusion creep.
         */
        double activation_volume_dislocation;

        /**
         * The prefactor used in the calculation of diffusion creep, including water fugacity,
         * water fugacity exponent, grain size and grain size exponent.
         */
        double prefactor_dislocation;

        /**
         * A scaling factor that can be used in the calculation of dislocation creep.
         */
        double nu_dislocation;

        /**
         * A scaling factor that can be used in the calculation of diffusion creep.
         */
        double nu_diffusion;

        /**
         * The exponent of strain rate used in the calculation of dislocation creep.
         */

        double stress_exponent;
        /**
         * The angle of internal friction used in the calculation of plasticity.
         */

        double phi;
        /**
         * The cohesion used in the calculation of plasticity.
         */
        double C;

        /**
         * The number of initial adaptive refinement cycles.
         */
        unsigned int initial_adaptive_refinement;

////////////////////////////////////////////////////////////////////////////////////////////////////
        double pTsm;
        double height_to_surface;
        double reference_compressibility;
////////////////////////////////////////////////////////////////////////////////////////////////////

    };

  }
}

#endif

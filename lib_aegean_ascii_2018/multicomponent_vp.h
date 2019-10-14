/*
  Copyright (C) 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__model_multicomponent_vp_h
#define __aspect__model_multicomponent_vp_h

#include <aspect/material_model/interface.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * A material model which is intended for use with multiple compositional
     * fields. Each compositional field is meant to be a single rock type,
     * where the value of the field at a point is interpreted to be a volume
     * fraction of that rock type.  If the sum of the compositional field
     * volume fractions is less than one, then the remainder of the volume is
     * assumed to be ``background mantle''.  If the sum of the compositional
     * field volume fractions is greater than one, then they are renormalized
     * to sum to one and there is no background mantle.
     *
     * For each material parameter the user supplies a comma delimited list of
     * length N+1, where N is the number of compositional fields.  The
     * additional field corresponds to the value for background mantle.  They
     * should be ordered ``background, composition1, composition2...''
     *
     * If a single value is given, then all the compositional fields are given
     * that value. Other lengths of lists are not allowed.  For a given
     * compositional field the material parameters are treated as constant,
     * except density, which varies linearly with temperature according to the
     * thermal expansivity.
     *
     * When more than one field is present at a point, they are averaged
     * arithmetically. An exception is viscosity, which may be averaged
     * arithmetically, harmonically, geometrically, or by selecting the
     * viscosity of the composition with the greatest volume fraction.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class MulticomponentVP : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
            MaterialModel::MaterialModelOutputs<dim> &out) const;
        /**
         * @name Physical parameters used in the basic equations
         * @{
         */

        /**
         * @}
         */

        /**
         * @name Qualitative properties one can ask a material model
         * @{
         */

        virtual bool
        viscosity_depends_on (const NonlinearDependence::Dependence dependence) const;

        virtual bool
        density_depends_on (const NonlinearDependence::Dependence dependence) const;

        virtual bool
        compressibility_depends_on (const NonlinearDependence::Dependence dependence) const;

        virtual bool
        specific_heat_depends_on (const NonlinearDependence::Dependence dependence) const;

        virtual bool
        thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * This model is not compressible, so this returns false.
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




        /**
         * @name Functions used in dealing with run-time parameters
         * @{
         */

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);
        /**
         * @}
         */

      private:
        double viscosity (const double                  temperature,
            const double                  pressure,
            const std::vector<double>    &compositional_fields,
            const SymmetricTensor<2,dim> &strain_rate,
            const Point<dim>             &position) const;

        double viscosity_ratio (const double                  temperature,
            const double                  pressure,
            const std::vector<double>    &compositional_fields,
            const SymmetricTensor<2,dim> &strain_rate,
            const Point<dim>             &position) const;

        double density (const double temperature,
            const double pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const;

        double compressibility (const double temperature,
            const double pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const;

        double specific_heat (const double temperature,
            const double pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const;

        double thermal_expansion_coefficient (const double      temperature,
            const double      pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const;

        double thermal_conductivity (const double temperature,
            const double pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const;

        double reference_density () const;

        double reference_thermal_expansion_coefficient () const;

        double reference_thermal_diffusivity () const;

        double reference_cp () const;


        /**
         * From a list of compositional fields of length N, we come up with an
         * N+1 length list that which also includes the fraction of
         * ``background mantle''. This list should sum to one, and is
         * interpreted as volume fractions.  If the sum of the
         * compositional_fields is greater than one, we assume that there is
         * no background mantle (i.e., that field value is zero).  Otherwise,
         * the difference between the sum of the compositional fields and 1.0
         * is assumed to be the amount of background mantle.
         */
         const std::vector<double> compute_volume_fractions(
             const std::vector<double> &compositional_fields) const;



         /**
          * Reference temperature for thermal expansion.  All components use
          * the same reference_T.
          */
         double reference_T;

         /**
          * Minimum viscosity cutoff
          */
         double minimum_visc;

         /**
          * Maximum viscosity cutoff
          */
         double maximum_visc;

         /**
          * Whether or not to smooth the transition from lower
          * to upper mantle viscosity
          */
         bool do_660_visc_smoothing;
         bool family_b;

         /**
          * Parsed function that specifies a prefactor for the plate boundary
          * effective viscosity.
          */
         //Functions::ParsedFunction<dim> viscosity_function;
         std::unique_ptr<Functions::ParsedFunction<dim> > viscosity_function;

         struct deformation
         {
             double n;
             double A_diff;
             double A_disl;
             double V_diff;
             double V_disl;
             double Q_diff;
             double Q_disl;
             double phi;
             double C;
             double n_Peierls;
             double A_Peierls;
             double V_Peierls;
             double Q_Peierls;
             double nu_diff;
             double nu_disl;
             double nu_Peierls;

             deformation() :
               n(),
               A_diff(),
               A_disl(),
               V_diff(),
               V_disl(),
               Q_diff(),
               Q_disl(),
               phi(),
               C(),
               n_Peierls(20),
               A_Peierls(1e-300),
               V_Peierls(10e-6),
               Q_Peierls(54e4),
               nu_diff(1),
               nu_disl(1),
               nu_Peierls(1)
             {}

             deformation(
                 double n,
                 double A_diff,
                 double A_disl,
                 double V_diff,
                 double V_disl,
                 double Q_diff,
                 double Q_disl,
                 double phi,
                 double C,
                 double n_Peierls,
                 double A_Peierls,
                 double V_Peierls,
                 double Q_Peierls,
                 double nu_diff,
                 double nu_disl,
                 double nu_Peierls)
             :
               n(n),

               A_diff(A_diff),
               A_disl(A_disl),
               V_diff(V_diff),
               V_disl(V_disl),
               Q_diff(Q_diff),
               Q_disl(Q_disl),
               phi(phi),
               C(C),
               n_Peierls(n_Peierls),
               A_Peierls(A_Peierls),
               V_Peierls(V_Peierls),
               Q_Peierls(Q_Peierls),
               nu_diff(nu_diff),
               nu_disl(nu_disl),
               nu_Peierls(nu_Peierls)
             {}
         };

         deformation wet_olivine, dry_olivine;
         deformation wet_quartz;
         deformation plag_anor_75;
         deformation all_wet_olivine;
         // Only diffusion, for e.g. the lower mantle
         deformation diffusion_only;
         // A constant, uniform Newtonian viscosity of 1e19, 1e20 or 1e22 Pas
         deformation constant19, constant20, constant21, constant22, constant24;
         // Small friction coefficient Mohr-Coulomb plasticity for subduction weak zone and boundary faults
         deformation mohr_coulomb_weak_zone, mohr_coulomb_wedge;

         struct material
         {
             double rho_0;
             double T_0;
             double k;
             double alpha;
             double c_P;

             material():
               rho_0(3250.0),
               T_0(293.5),
               k(2.0),
               alpha(2e-5),
               c_P(1250.0) {}

             material(
                 double rho_0,
                 double T_0,
                 double k,
                 double alpha,
                 double c_P):
                   rho_0(rho_0),
                   T_0(T_0),
                   k(k),
                   alpha(alpha),
                   c_P(c_P) {}
         };

         material oceanic_crust, continental_crust, upper_mantle, transition_mantle, lower_mantle, sticky_air, ocean;

         std::vector<material> material_types;
         mutable std::vector<deformation> deformation_types;
         deformation lower_mantle_deformation_type;
         deformation upper_mantle_deformation_type;


         /*
          * Function to calculate diffusion creep.
          */
         double diffusion(const deformation def,
             const double temperature,
             const double pressure) const;
         /*
          * Function to calculate dislocation creep.
          */
         double dislocation(const deformation def,
             const double temperature,
             const double pressure,
             const double strain_rate) const;

         /*
          * Function to calculate Peierls creep.
          */
         double Peierls(const deformation def,
             const double temperature,
             const double pressure,
             const double strain_rate) const;

         /*
          * Function to calculate plasticity.
          */
         double plastic(const deformation def,
             const double pressure,
             const double strain_rate) const;




         /**
          * Enumeration for selecting which viscosity averaging scheme to use.
          * Select between harmonic, arithmetic, geometric, and
          * maximum_composition.  The max composition scheme simply uses the
          * viscosity of whichever field has the highes volume fraction.
          */
         enum
         {
           harmonic,
           arithmetic,
           geometric,
           maximum_composition
         } viscosity_averaging;


         /**
          * Vector for field initial viscosities, read from parameter file.
          */
         std::vector<double> initial_viscosities;


         /**
          * Number of compositions, including the mantle
          */
         unsigned int n_fields;

         /**
          * Number of compositions, excluding the mantle
          */
         unsigned int n_compositional_fields;

         //      public:
         /**
          * @}
          */
         //        virtual
         //        double get_alpha(std::string type) const;


    };

  }
}

#endif

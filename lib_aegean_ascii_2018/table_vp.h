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


#ifndef __aspect__model_table_vp_h
#define __aspect__model_table_vp_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    namespace internal
    {
      class MaterialLookup;
    }
    /**
     * A variable viscosity material model that reads the essential values of
     * coefficients from tables in input files.
     *
     * The viscosity of this model is based on the paper
     * Steinberger/Calderwood 2006: "Models of large-scale viscous flow in the
     * Earth's mantle with contraints from mineral physics and surface
     * observations". The thermal conductivity is constant and the other
     * parameters are provided via lookup tables from the software PERPLEX.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class TableVp: public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:

        /**
         * Initialization function. Loads the material data and sets up
         * pointers.
         */
        virtual
        void
        initialize ();

        /**
         * @name Physical parameters used in the basic equations
         * @{
         */
        virtual double viscosity (const double                  temperature,
                                  const double                  pressure,
                                  const std::vector<double>    &compositional_fields,
                                  const SymmetricTensor<2,dim> &strain_rate,
                                  const Point<dim>             &position) const;

//        virtual double viscosity_ratio (const double                  temperature,
//                                  const double                  pressure,
//                                  const std::vector<double>    &compositional_fields,
//                                  const SymmetricTensor<2,dim> &strain_rate,
//                                  const Point<dim>             &position) const;

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

        virtual double thermal_conductivity (const double temperature,
                                             const double pressure,
                                             const std::vector<double> &compositional_fields,
                                             const Point<dim> &position) const;

        virtual double thermal_expansion_coefficient (const double      temperature,
                                                      const double      pressure,
                                                      const std::vector<double> &compositional_fields,
                                                      const Point<dim> &position) const;

        virtual double seismic_Vp (const double      temperature,
                                   const double      pressure,
                                   const std::vector<double> &compositional_fields,
                                   const Point<dim> &position) const;

        virtual double seismic_Vs (const double      temperature,
                                   const double      pressure,
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

        virtual double reference_specific_heat () const;

        virtual double reference_thermal_diffusivity () const;

        virtual double reference_thermal_conductivity () const;



        /**
         * @}
         */

        /**
         * Function to compute the material properties in @p out given the
         * inputs in @p in. If MaterialModelInputs.strain_rate has the length
         * 0, then the viscosity does not need to be computed.
         */
        virtual
        void
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                 MaterialModel::MaterialModelOutputs<dim> &out) const;

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
        bool interpolation;
        bool latent_heat;
        bool compressible;
        double reference_eta;
        std::string datadirectory;
        std::vector<std::string> material_file_names;
        unsigned int n_material_data;

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
         * In the incompressible case we need to adjust the temperature as if
         * there would be an adiabatic temperature increase to look up the
         * material properties in the lookup table.
         */
        double get_corrected_temperature (const double temperature,
                                          const double pressure,
                                          const Point<dim> &position) const;

        /**
         * In the incompressible case we need to adjust the pressure as if
         * there would be an compressible adiabatic pressure increase to look
         * up the material properties in the lookup table. Unfortunately we do
         * not know the adiabatic pressure profile for the incompressible case
         * and therefore we do not know the dynamic pressure. The only
         * currently possible solution is to use the adiabatic pressure
         * profile only, neglecting dynamic pressure for material lookup in
         * this case. This is essentially similar to having a depth dependent
         * reference profile for all properties and modifying the profiles
         * only in temperature-dimension.
         */
        double get_corrected_pressure (const double temperature,
                                       const double pressure,
                                       const Point<dim> &position) const;

        /**
         * This function returns the compressible density derived from the
         * list of loaded lookup tables.
         */
        double get_compressible_density (const double temperature,
                                         const double pressure,
                                         const std::vector<double> &compositional_fields,
                                         const Point<dim> &position) const;

        /**
         * We need to correct the compressible density in the incompressible
         * case to an incompressible profile. This is done by dividing the
         * compressible density with the density at this pressure at adiabatic
         * temperature and multiplying with the surface adiabatic density.
         */
        double get_corrected_density (const double temperature,
                                      const double pressure,
                                      const std::vector<double> &compositional_fields,
                                      const Point<dim> &position) const;

        /**
         * List of pointers to objects that read and process data we get from
         * Perplex files. There is one pointer/object per compositional field
         * data provided.
         */
        std::vector<std::shared_ptr<internal::MaterialLookup> > material_lookup;

        mutable std::vector<unsigned int> material_per_composition;

        unsigned int n_compositional_fields;
        unsigned int n_fields;

        /**
         * Pointer to the material model used as the base model
         */
        std::shared_ptr<MaterialModel::Interface<dim> > viscosity_model;


    };
  }
}

#endif

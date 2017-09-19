/*
  Copyright (C) 2014 - 2016 by the authors of the ASPECT code.

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



#ifndef _aspect_initial_temperature_lithosphere_rift_h
#define _aspect_initial_temperature_lithosphere_rift_h

#include <aspect/initial_temperature/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{


  namespace InitialTemperature
  {

    /**
     *
     * @ingroup InitialTemperatures
     */
    template <int dim>
    class LithosphereRift : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        LithosphereRift ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        void
        initialize ();

        /**
         * Return the initial temperature as a function of position.
         */
        virtual
        double initial_temperature (const Point<dim> &position) const;

        /**
         * Return the initial temperature as a function of depth and
         * the local layer thicknesses.
         */
        virtual
        double temperature (const double depth,
                            const std::vector<double> thicknesses) const;

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

      private:
        /**
         * Surface temperature
         */
        double T0;

        /**
         * LAB isotherm temperature
         */
        double LAB_isotherm;

        /**
         * Vector for field heat production rates.
         */
        std::vector<double> heat_productivities;

        /**
         * Vector for thermal conductivities.
         */
        std::vector<double> conductivities;

        /**
         * Vector for field densities.
         */
        std::vector<double> densities;

        /**
         * Vector for the reference field thicknesses.
         */
        std::vector<double> thicknesses;

        /**
         * The standard deviation of the Gaussian amplitude of the lithospheric thicknesses.
         */
        double sigma;
    };
  }
}


#endif

/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#ifndef __aspect__initial_temperature_model_ridge_segments_h
#define __aspect__initial_temperature_model_ridge_segments_h

#include <aspect/initial_temperature/interface.h>
#include <aspect/simulator.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace InitialTemperature
  {
    using namespace dealii;

    /**
     * A class that describes the temperature field according to the plate cooling model
     *
     * @ingroup InitialConditionsModels
     */
    template <int dim>
    class RidgeSegments : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        RidgeSegments ();

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

      private:
        /**
         *The parameters needed for the plate cooling model
         */

        /**
         * The spreading velocity of the plates to calculate their age
         * with distance from the trench
         * TODO at some point we could use different velocities for
         * each segment
         */
        double spreading_velocity;

        /**
         * The maximum thickness and temperature of an oceanic plate
         * when time goes to infinity
         */
        double max_plate_thickness;
        double Tm;

        /**
         * The temperature at the top boundary of the domain
         */
        double Ts;

        /**
         * The thermal diffusivity used in the computation
         * of the plate cooling model
         */
        double thermal_diffusivity;

    };
  }
}

#endif

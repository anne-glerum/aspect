/*
  Copyright (C) 2012 by the authors of the ASPECT code.

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


#ifndef __aspect__initial_temperature_adiabat_ascii_h
#define __aspect__initial_temperature_adiabat_ascii_h

#include <aspect/initial_temperature/interface.h>
#include <aspect/simulator.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace InitialTemperature
  {
    using namespace dealii;

    /**
     * A class that implements adiabatic initial conditions for the
     * temperature field and, optional, upper and lower thermal boundary
     * layers calculated using the half-space cooling model. The age of the
     * boundary layers are input parameters.
     *
     * @ingroup InitialTemperatureModels
     */
    template <int dim>
    class AdiabatAscii : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
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
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        /**
         * Age of the upper thermal boundary layer at the surface of the
         * model. If set to zero, no boundary layer will be present in the
         * model.
         */
        double age_top_boundary_layer;
        /* Age of the lower thermal boundary layer. */
        double age_bottom_boundary_layer;
        double thickness_top;
        double thickness_bottom;

        /*
         * Whether or not to add temperature perturbations
         */
        bool add_perturbation;

        /*
         * Whether or not to smooth the temperature perturbations
         */
        bool add_top_smoothing;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double top_smoothing_depth;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double top_smoothing_width;

        /*
         * Whether or not to smooth the temperature perturbations
         */
        bool add_bottom_smoothing;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double bottom_smoothing_depth;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double bottom_smoothing_width;

        /*
         * Whether or not to smooth the temperature perturbations
         */
        bool add_boundary_smoothing;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double boundary_smoothing_angle;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double boundary_smoothing_anglewidth;

        /*
         * Whether or not to smooth the temperature perturbations
         */
        bool add_slab_smoothing;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double slab_smoothing_angle;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double slab_smoothing_depth;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double slab_smoothing_anglewidth;

        /*
         * The depth around which to smooth the temperature perturbations
         */
        double slab_smoothing_width;

        /*
         * Deviation from adiabaticity in a subadiabatic mantle
         * temperature profile. 0 for an adiabatic temperature
         * profile.
         */
        double subadiabaticity;

        // TODO: make this pointer to ascii data plugin immediately
        std::shared_ptr<InitialTemperature::Interface<dim> > ascii_model;
    };
  }
}


#endif

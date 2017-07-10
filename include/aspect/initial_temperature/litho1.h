/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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


#ifndef _aspect_initial_temperature_litho1_h
#define _aspect_initial_temperature_litho1_h

//#include <aspect/initial_temperature/adiabatic.h>
#include <aspect/initial_temperature/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace InitialTemperature
  {
    using namespace dealii;

    /**
     * A class that implements a prescribed temperature field determined from
     * a AsciiData input file.
     *
     * @ingroup InitialTemperatures
     */
    template <int dim>
    class Litho1 : public Utilities::AsciiDataBoundary<dim>,  public Interface<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        Litho1 ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        void
        initialize ();

        // avoid -Woverloaded-virtual:
        using Utilities::AsciiDataBoundary<dim>::initialize;

        /**
         * Return the boundary temperature as a function of position. For the
         * current class, this function returns value from the text files.
         */
        double
        initial_temperature (const Point<dim> &position) const;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm);

      private:
        /*
         * The boundary indicartor that represents
         * the surface of the domain.
         */
        types::boundary_id surface_boundary_id;
  
        /*
         * The isotherm that is to represent the LAB.
         * Below this isotherm an continental temperature
         * profile is prescribed, below an adiabatic profile.
         * TODO: when using other plugins, how to match the 
         * temperatures at the LAB?
         */
        double LAB_isotherm;

       /*
        * The temperature at the model's surface.
        */
       double T0;

       /*
        * The depth to which the LAB isotherm temperature is
        * prescribed below the LAB depth.
        */
       double compensation_depth;
    };
  }
}


#endif

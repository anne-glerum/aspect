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

        /**
         * The function that computes a steady-state temperature profile
         * in the lithosphere including the user-specified radioactive heating.
         */
        double continental_geotherm (const double depth,
                                     const std::vector<double> thicknesses) const;

        /*
         * The boundary indicartor that represents
         * the surface of the domain.
         */
        types::boundary_id surface_boundary_id;
        types::boundary_id bottom_boundary_id;
        types::boundary_id left_boundary_id;

        /*
         * The isotherm that is to represent the LAB.
         * Above this isotherm an continental temperature
         * profile is prescribed, below an adiabatic profile.
         * TODO: when using other plugins, how to match the
         * temperatures at the LAB?
         */
        double LAB_isotherm;

        /**
         * The minimum thickness of the lithosphere
         * regardless of the thickness from the ascii table.
         */
        double min_LAB_thickness;


        /**
         * Whether or not to merge two overlapping data sets for
         * the LAB thickness.
         */
         bool merge_LAB_grids;

         /**
          * The halfwidth of the hyperbolic tangent to merge
          * the two LAB gridth with.
          */
         double merge_LAB_grids_halfwidth;

         /**
          * The list of polygon points for smoothing.
          */
         std::vector<Point<2> > polygon_point_list;

        /*
         * The temperature at the model's top boundary.
         */
        double T0;

        /**
         * The fraction of the crust that will be
         * designated as upper crust.
         */
        double upper_crust_fraction;

        /*
         * The depth to which the LAB isotherm temperature is
         * prescribed below the LAB depth.
         */
        double compensation_depth;

        /**
         * Vector for field heat production rates, read from parameter file .
         */
        std::vector<double> heat_productivities;

        /**
         * Vector for thermal conductivities, read from parameter file .
         */
        std::vector<double> conductivities;

        /**
         * Vector for field densities, read from parameter file .
         */
        std::vector<double> densities;

    };
  }
}


#endif

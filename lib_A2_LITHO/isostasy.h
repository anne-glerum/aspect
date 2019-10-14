/*
  Copyright (C) 2016 by the authors of the ASPECT code.

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
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#ifndef _aspect_geometry_model__initial_topography_isostasy_h
#define _aspect_geometry_model__initial_topography_isostasy_h

#include <aspect/geometry_model/initial_topography_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace InitialTopographyModel
  {
    using namespace dealii;

    /**
     * A class that describes an initial topography for the geometry model,
     * by defining a set of polylines on the surface from the prm file. It
     * sets the elevation in each Polyline to a constant value.
     */
    template <int dim>
    class Isostasy : public Interface<dim>, public Utilities::AsciiDataBoundary<dim>
    {
      public:
        // Empty constructor
        Isostasy ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        void
        initialize ();

        // avoid -Woverloaded-virtual:
        using Utilities::AsciiDataBoundary<dim>::initialize;

        /**
         * Return the value of the topography for a point.
         */
        virtual
        double value (const Point<dim-1> &p) const;

        /**
         * Return the maximum value of the elevation.
         */
        virtual
        double max_topography () const;

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
         * The boundary indicators that represent
         * the surface and bottom of the domain.
         */
        types::boundary_id surface_boundary_id;
        types::boundary_id bottom_boundary_id;
        types::boundary_id left_boundary_id;

        /**
         * Vector for the reference field thicknesses
         * which are used for isostasy calculations.
         */
        std::vector<double> thicknesses;

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

        /**
         * The fraction of the crust that will be
         * designated as upper crust instead of reading
         * it's thickness from the ascii data file.
         */
        double upper_crust_fraction;

        /**
         * The reference lithospheric column used in computing the topography based on isostasy
         * and the thickness of this column.
         */
        double ref_rgh;
        double compensation_depth;

        /**
         * Vector for field densities.
         */
        std::vector<double> densities;
        std::vector<double> temp_densities;

        /**
         * The maximum topography in this model
         */
        double maximum_topography;

    };
  }
}


#endif

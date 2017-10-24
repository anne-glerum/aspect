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


#ifndef __aspect__initial_composition_ridge_segments_h
#define __aspect__initial_composition_ridge_segments_h

#include <aspect/initial_composition/interface.h>
#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace InitialComposition
  {
    using namespace dealii;

    /**
     * A class that describes the compositional fields according to the plate cooling model
     *
     * @ingroup CompositionInitialConditionsModels
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
         * Return the initial composition as a function of position.
         */
        virtual
        double initial_composition (const Point<dim> &position,
                                    const unsigned int n_comp) const;

       /**
        * Compute the distance to the ridge.
        */
        double distance_to_ridge (const Point<2> &position,
                                 const bool cartesian_geometry) const;

       /**
        * Return the surface coordinate.
        */
        Point<2> surface_position (const Point<dim> &position,
                                   const bool cartesian_geometry) const;

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
         */
        double spreading_velocity;

        /**
         * The constant thickness of the oceanic crust
         */
        double crustal_thickness;

        /**
         * The maximum thickness of an oceanic plate
         * when time goes to infinity
         */
        double max_plate_thickness;

       /**
        * The thermal diffusivity used in the computation
        * of the plate cooling model
        */
        double thermal_diffusivity;

        /**
         * The composition numbers of the crust and mantle lithosphere.
         */
        unsigned int id_mantle_L;
        unsigned int id_crust;

        /**
         * The list of line segments consisting of two 2D coordinates per segment.
         * The segments represent the rift axis.
         */
        std::vector<std::vector<Point<2> > > point_list;

    };
  }
}

#endif

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


#ifndef _aspect_initial_composition_litho1_h
#define _aspect_initial_composition_litho1_h

#include <aspect/initial_composition/interface.h>

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace InitialComposition
  {
    using namespace dealii;

    /**
     * A class that implements the prescribed compositional fields determined
     * from a AsciiData input file.
     *
     * @ingroup InitialCompositionModels
     */
    template <int dim>
    class Litho1 : public Interface<dim>, public Utilities::AsciiDataBoundary<dim>
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
         * Return the initial composition as a function of position. For the
         * current class, this function returns value from the text files.
         */
        double
        initial_composition (const Point<dim> &position,
                             const unsigned int n_comp) const;

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
         * The boundary indicators that represent
         * the surface and bottom of the domain.
         */
        types::boundary_id surface_boundary_id;
        types::boundary_id bottom_boundary_id;

        /**
         * The fraction of the crust that will be
         * designated as upper crust instead of reading
         * it's thickness from the ascii data file.
         */
        double upper_crust_fraction;

        /**
         * The minimum thickness of the lithosphere
         * regardless of the thickness from the ascii table.
         */
        double min_LAB_thickness;

        /**
         * The compositional field number of the lower crust.
         * For an upper and lower crust, the upper crust
         * will be field 0, the lower crust field 1 and
         * the lithospheric mantle field 2. When the upper
         * crust fraction is set to zero and no compositional
         * field called "upper" is present, the crust field
         * will be field 0 and the lithospheric mantle field 1.
         */
        unsigned int lower_crust_id;
        unsigned int upper_crust_id;
        unsigned int mantle_L_id;

    };
  }
}


#endif

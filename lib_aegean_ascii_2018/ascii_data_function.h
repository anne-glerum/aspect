/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__initial_composition_ascii_data_function_h
#define __aspect__initial_composition_ascii_data_function_h

#include <aspect/initial_composition/interface.h>
#include <deal.II/base/parsed_function.h>

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace InitialComposition
  {
    using namespace dealii;

    /**
     * A class that implements the prescribed compositional fields determined
     * from a AsciiDataFunction input file.
     *
     * @ingroup InitialCompositionModels
     */
    template <int dim>
    class AsciiDataFunction : public Interface<dim>, public Utilities::AsciiDataInitial<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        AsciiDataFunction ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        void
        initialize ();

        // avoid -Woverloaded-virtual:
        using Utilities::AsciiDataInitial<dim>::initialize;

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
         *Whether or not to construct a gap in the slab
         */ 
         bool slab_gap;

        /**
         * The field number of the slab field
         */ 
         double n_slab;

         /**
          * A function object representing the gap in the slab.
          */
         Functions::ParsedFunction<dim> function;

         /*
          * Pointer to base model
          */  
         std::shared_ptr<InitialComposition::Interface<dim> > ascii_model;

         /*
          * Radius at which to switch from upper to lower ascii table.
          */
          double switch_table_radius; 
    };
  }
}


#endif

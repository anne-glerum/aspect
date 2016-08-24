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
/*  $Id: function.h 1509 2012-12-12 05:50:56Z bangerth $  */


#ifndef __aspect__compositional_initial_conditions_function4_h
#define __aspect__compositional_initial_conditions_function4_h

#include <aspect/compositional_initial_conditions/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace CompositionalInitialConditions
  {
    using namespace dealii;

    /**
     * A class that implements initial conditions for the
     * compositional fields based on a
     * functional description provided in the input file.
     *
     * @ingroup CompositionalInitialConditionsModels
     */
    template <int dim>
    class Subduction4 : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */

        /**
         * Return the initial composition as a function of position and number of compositional field.
         */
        virtual
        double initial_composition (const Point<dim> &position, const unsigned int n_comp) const;

        /**
         * Declare the parameters this class takes through input files.
         * The default implementation of this function does not describe
         * any parameters. Consequently, derived classes do not have to
         * overload this function if they do not take any runtime parameters.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter
         * file. The default implementation of this function does not read
         * any parameters. Consequently, derived classes do not have to
         * overload this function if they do not take any runtime parameters.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
	bool b_Use_sticky_air;
        int i_Subducting_crust_layers;
        int i_Overrding_crust_layers;
	double composition_value;
	double Height_to_surface; 			/// depth
	double Initial_subduction_point; 		/// Px
	double Slab_dip_rad;
        double Slab_dip; 				/// dip
        double Slab_length; 				/// d
        double Crustal_thickness_subducting_plate; 	/// lcr2
        double Lithospheric_thickness_subducting_plate; /// l
        double Crustal_thickness_overriding_plate; 	/// lcr3/lc3cr
        double Lithospheric_thickness_overriding_plate; /// lc3
	double Radius_smoothing_cricle; 		/// r
	int n_compositional_fields;

	unsigned int                   wkz_n_zones;
        std::vector<double>            wkz_composition;
        std::vector<double>            wkz_orderx;
        std::vector<double>            wkz_ordery;
        std::vector<double>            wkz_length;
        std::vector<double>            wkz_angle;
	std::vector<double>            wkz_thickness;
	std::vector<std::string>       wkz_feature;
	std::vector<double>            wkz_upperboundary;
	std::vector<double>            wkz_lowerboundary;
	std::vector<double>            wkz_rotationpoint;
	std::vector<std::vector<double> > wkz_beginpoints;

////////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<double>	       transition_depths;
	std::vector<double>	       Clapeyron_slopes;
	std::vector<double>	       transition_widths;
	std::vector<double>	       transition_pressures;

	double reference_specific_heat;
	double thermal_alpha;
	double pTsm;
////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * A function object representing the compositional fields.
         */
        std::auto_ptr<Functions::ParsedFunction<dim> > function;
    };
  }
}


#endif

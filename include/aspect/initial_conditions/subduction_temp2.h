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
/*  $Id: mckenzie.h 1088 2013-10-22 13:46 glerum $  */


#ifndef __aspect__initial_conditions_subduction_temp2_h
#define __aspect__initial_conditions_subduction_temp2_h

#include <aspect/initial_conditions/interface.h>
#include <aspect/simulator.h>

#include <deal.II/base/parsed_function.h>


namespace aspect
{
  namespace InitialConditions
  {
    using namespace dealii;

    /**
     * A class that implements mckenzie initial conditions
     * for the temperature field and, optional, upper and
     * lower thermal boundary layers calculated
     * using the half-space cooling model. The age of the
     * boundary layers are input parameters.
     *
     * @ingroup InitialConditionsModels
     */
    template <int dim>
    class SubductionTemp2 : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
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
         * Read the parameters this class declares from the parameter
         * file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:

         double l;
         double coordrot;
         double dip;
         double Px;
         double d;
         double dwz;
         double lc3;
         double lcr2;
         double v; 
         double pTsm;
         double n_sum;
//Please uncomment the following when the inputfile is correct
         double Cp;
         double k;
         double alfa;
         double density;
//////
	 
	unsigned int                   wkz_n_zones;
        std::vector<double>            wkz_heating;
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
	 
         //double g;
         double depth;
         double T_pot;
	 double Ttop;
        /**
         * A function object representing the compositional fields
         * that will be used as a reference profile for calculating
         * the thermal diffusivity.
         * The function depends only on depth.
         */
        std::auto_ptr<Functions::ParsedFunction<1> > function;
    };
  }
}


#endif

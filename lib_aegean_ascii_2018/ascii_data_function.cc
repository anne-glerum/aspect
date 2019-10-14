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


#include <aspect/global.h>
#include <aspect/initial_composition/ascii_data.h>
#include "ascii_data_function.h"


namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    AsciiDataFunction<dim>::AsciiDataFunction ()
    {}


    template <int dim>
    void
    AsciiDataFunction<dim>::initialize ()
    {
      Utilities::AsciiDataInitial<dim>::initialize(this->n_compositional_fields());
    }


    template <int dim>
    double
    AsciiDataFunction<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int n_comp) const
    {
      double field_value = 0.;

       const std::array<double,dim> spherical_coordinates =
                        aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
       Point<dim> point;
       for (unsigned int i = 0; i<dim; ++i)
           point[i] = spherical_coordinates[i];
       // Scale longitude to [-pi,pi]
       if (point[1] > numbers::PI)
          point[1] -= 2.0*numbers::PI;

       // Below this radius, every compositional field will be zero
       // anyway, so no need to look up the value in a table.
       if (point[0] < 5571000.0)
         return 0.;

//      if (position.norm() < switch_table_radius)
//      if (position.norm() < switch_table_radius || point[1]<-16.9/180.0*numbers::PI || point[1]>15./180.0*numbers::PI || point[2]<80./180.0*numbers::PI || point[2]>100./180.0*numbers::PI)
      // Use one table when within specified bounds, otherwise the second table.      
      if (point[1]<-16.9/180.0*numbers::PI || point[1]>15./180.0*numbers::PI || point[2]<80./180.0*numbers::PI || point[2]>100./180.0*numbers::PI)
       //field_value = ascii_model->initial_composition(position, n_comp);
       field_value = this->get_initial_composition_manager().template get_matching_initial_composition_model<InitialComposition::AsciiData<dim> >().initial_composition(position, n_comp);
      else
       field_value = Utilities::AsciiDataInitial<dim>::get_data_component(position,n_comp);

      // If a slab gap is wanted and the current compositional field
      // represents the slab, adjust the compositional value.
      if (slab_gap && n_comp == n_slab)
      {
       field_value = std::min(function.value(point),field_value);
      }
      return field_value;
    }


    template <int dim>
    void
    AsciiDataFunction<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Ascii data with function");
        {
         prm.declare_entry("Construct slab gap",
                           "false", Patterns::Bool (),
                           "Whether or not to add a gap in the slab, as specified in the bounds file. ");
         prm.declare_entry("Composition number slab",
                           "0", Patterns::Double (0),
                           "If a slab gap is specified, for which compositional field number? ");
         Functions::ParsedFunction<dim>::declare_parameters (prm, 1);

         Utilities::AsciiDataBase<dim>::declare_parameters(prm,
                                                          "$ASPECT_SOURCE_DIR/data/compositional-initial-conditions/ascii-data/test/",
                                                          "box_2d.txt");
          prm.declare_entry("Base model","ascii data",
                            Patterns::Selection("ascii data"),
                            "The name of a composition initial model that will be modified "
                            "by the current model. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Initial composition model models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry("Ascii table merge radius", 
                            "6171000", Patterns::Double (0),
                            "At what radius is the switch from lower to upper ascii table made? ");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    AsciiDataFunction<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Ascii data with function");
        {
         Utilities::AsciiDataBase<dim>::parse_parameters(prm);
         slab_gap = prm.get_bool("Construct slab gap");
         n_slab   = prm.get_double("Composition number slab");
         try
           {
             function.parse_parameters (prm);
           }
         catch (...)
           {
             std::cerr << "ERROR: FunctionParser failed to parse\n"
                       << "\t'Initial composition model.Function'\n"
                       << "with expression\n"
                       << "\t'" << prm.get("Function expression") << "'\n"
                       << "More information about the cause of the parse error \n"
                       << "is shown below.\n";
             throw;
           }

        AssertThrow( prm.get("Base model") != "ascii data function",
                     ExcMessage("You may not use ``ascii data function'' as the base model for "
                                "a this composition initial conditions model.") );
          //ascii_model.reset(InitialComposition::create_initial_conditions<dim>(prm.get("Base model")));
          //if ( SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(ascii_model.get()))
          //  sim->initialize_simulator (this->get_simulator());
         switch_table_radius = prm.get_double("Ascii table merge radius");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // parse the ascii model parameters
      //ascii_model->parse_parameters(prm);
      //ascii_model->initialize();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(AsciiDataFunction,
                                                     "ascii data function",
                                                     "Implementation of a model in which the initial "
                                                     "composition is derived from files containing data "
                                                     "in ascii format. Note the required format of the "
                                                     "input data: The first lines may contain any number of comments "
                                                     "if they begin with '#', but one of these lines needs to "
                                                     "contain the number of grid points in each dimension as "
                                                     "for example '# POINTS: 3 3'. "
                                                     "The order of the data columns "
                                                     "has to be 'x', 'y', 'composition1', 'composition2', "
                                                     "etc. in a 2d model and 'x', 'y', 'z', 'composition1', "
                                                     "'composition2', etc. in a 3d model, according "
                                                     "to the number of compositional fields, which means that "
                                                     "there has to be a single column "
                                                     "for every composition in the model."
                                                     "Note that the data in the input "
                                                     "files need to be sorted in a specific order: "
                                                     "the first coordinate needs to ascend first, "
                                                     "followed by the second and the third at last in order to "
                                                     "assign the correct data to the prescribed coordinates. "
                                                     "If you use a spherical model, "
                                                     "then the data will still be handled as Cartesian, "
                                                     "however the assumed grid changes. 'x' will be replaced by "
                                                     "the radial distance of the point to the bottom of the model, "
                                                     "'y' by the azimuth angle and 'z' by the polar angle measured "
                                                     "positive from the north pole. The grid will be assumed to be "
                                                     "a latitude-longitude grid. Note that the order "
                                                     "of spherical coordinates is 'r', 'phi', 'theta' "
                                                     "and not 'r', 'theta', 'phi', since this allows "
                                                     "for dimension independent expressions.")
  }
}

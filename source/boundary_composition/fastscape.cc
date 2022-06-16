/*
  Copyright (C) 2011 - 2020 by the authors of the ASPECT code.

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


#include <aspect/boundary_composition/fastscape.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/mesh_deformation/fastscape.h>

namespace aspect
{
  namespace BoundaryComposition
  {
// ------------------------------ FastScape -------------------
    template <int dim>
    FastScape<dim>::FastScape()
    {}

    template <int dim>
    void
    FastScape<dim>::update()
    {

      if (this->simulator_is_past_initialization() && this->get_timestep_number() == 0)
        {
          const std::set<types::boundary_id> boundary_ids
            = this->get_mesh_deformation_handler().get_active_mesh_deformation_boundary_indicators();
          std::map<types::boundary_id, std::vector<std::string>> mesh_deformation_boundary_indicators_map
                                                              = this->get_mesh_deformation_handler().get_active_mesh_deformation_names();

          bool using_fastscape = false;
          // Loop over each mesh deformation boundary, and check FastScape is used.
          for (std::set<types::boundary_id>::const_iterator p = boundary_ids.begin();
               p != boundary_ids.end(); ++p)
            {
              const std::vector<std::string> names = mesh_deformation_boundary_indicators_map[*p];
              for (unsigned int i = 0; i < names.size(); ++i)
                {
                  if (names[i] == "fastscape")
                    using_fastscape = true;
                }
            }

          AssertThrow(using_fastscape, ExcMessage("The boundary composition plugin FastScape requires the mesh deformation plugin FastScape. "));
          AssertThrow(this->introspection().compositional_name_exists("ratio_marine_continental_sediment"),
                      ExcMessage("The boundary composition plugin FastScape requires a compositional field called ratio_marine_continental_sediment."));
          AssertThrow(this->introspection().compositional_name_exists("silt_fraction"),
                      ExcMessage("The boundary composition plugin FastScape requires a compositional field called silt_fraction."));
        }
    }

    template <int dim>
    double
    FastScape<dim>::
    boundary_composition(const types::boundary_id boundary_indicator,
                         const Point<dim> &position,
                         const unsigned int compositional_field) const
    {
      // FastScape is only run in timestep 1.
      // The boundary_composition function is evaluated before mesh deformation,
      // and thus we can only question the FastScape mesh deformation plugin for
      // the sediment ratio in timestep 2.
      if (!this->simulator_is_past_initialization() || this->get_timestep_number() < 2)
        return 0.;

      const types::boundary_id top_boundary = this->get_geometry_model().translate_symbolic_boundary_name_to_id ("top");
      const unsigned int marine_ratio_field = this->introspection().compositional_index_for_name("ratio_marine_continental_sediment");
      const unsigned int silt_fraction_field = this->introspection().compositional_index_for_name("silt_fraction");

      // Only set composition on the top boundary,
      // and only for the field 'ratio_marine_continental_sediment'
      if (boundary_indicator != top_boundary || (compositional_field != marine_ratio_field && compositional_field != silt_fraction_field))
        return 0.;

      double result = 0.;

      const std::map<types::boundary_id,std::vector<std::unique_ptr<aspect::MeshDeformation::Interface<dim>>>> &mesh_deformation_objects
        = this->get_mesh_deformation_handler().get_active_mesh_deformation_models();

      // Instead of looping over all boundaries, directly select top boundary
      for (const auto &boundary_and_deformation_objects : mesh_deformation_objects)
        {
          if (boundary_and_deformation_objects.first == top_boundary)
            {
              for (const auto &model : boundary_and_deformation_objects.second)
                if (Plugins::plugin_type_matches<const MeshDeformation::FastScape<dim>>(*model))
                  {
                    if (compositional_field == marine_ratio_field)
                      result = Plugins::get_plugin_as_type<const MeshDeformation::FastScape<dim>>(*model).get_marine_to_continental_sediment_ratio(position);
                    else if (compositional_field == silt_fraction_field)
                      result = Plugins::get_plugin_as_type<const MeshDeformation::FastScape<dim>>(*model).get_silt_fraction(position);
                  }
            }
        }

      return result;
    }

    template <int dim>
    double
    FastScape<dim>::
    minimal_composition (const std::set<types::boundary_id> &) const
    {
      return min_composition;
    }



    template <int dim>
    double
    FastScape<dim>::
    maximal_composition (const std::set<types::boundary_id> &) const
    {
      return max_composition;
    }



    template <int dim>
    void
    FastScape<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary composition model");
      {
        prm.enter_subsection("FastScape");
        {
          prm.declare_entry ("Minimal composition", "0.",
                             Patterns::Double (),
                             "Minimal composition. Units: none.");
          prm.declare_entry ("Maximal composition", "1.",
                             Patterns::Double (),
                             "Maximal composition. Units: none.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    FastScape<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary composition model");
      {
        prm.enter_subsection("FastScape");
        {
          min_composition = prm.get_double ("Minimal composition");
          max_composition = prm.get_double ("Maximal composition");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryComposition
  {
    ASPECT_REGISTER_BOUNDARY_COMPOSITION_MODEL(FastScape,
                                               "fastscape",
                                               "A model in which the composition at the boundary "
                                               "for two specific field called 'marine_continental_ratio' "
                                               "and 'silt_fraction' are set by the FastScape mesh deformation plugin. ")
  }
}

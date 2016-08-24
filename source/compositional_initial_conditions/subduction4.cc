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
/*  $Id: function.cc 1389 2012-11-28 14:08:16Z bangerth $  */


#include <aspect/compositional_initial_conditions/subduction4.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/box.h>
//##include <aspect/geometry_model/asymbox.h>
#include <aspect/geometry_model/spherical_shell.h>
//#include <aspect/geometry_model/wgs84.h>
//#include <aspect/geometry_model/africa.h>

namespace aspect
{
  namespace CompositionalInitialConditions
  {

//    template <int dim>
//    Function<dim>::Function ()
//    {}

    template <int dim>
    double
    Subduction4<dim>::
    initial_composition (const Point<dim> &position, const unsigned int n_comp) const
    {
    	 // This initial condition only makes sense if the geometry is a
    	        // box. verify that it is indeed a box
//    	          const GeometryModel::AsymBox<dim> 			*geometry = 	dynamic_cast<const GeometryModel::AsymBox<dim>*> 		(&this->get_geometry_model());
    	          const GeometryModel::Box<dim> 				*geometryb = 	dynamic_cast<const GeometryModel::Box<dim>*> 			(&this->get_geometry_model());
    	          const GeometryModel::SphericalShell<dim> 		*geometryc = 	dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model());
  //  	          const GeometryModel::WGS84<dim> 				*geometryd = 	dynamic_cast<const GeometryModel::WGS84<dim>*> 			(&this->get_geometry_model());
    	          //const GeometryModel::Africa<dim> 				*geometrye = 	dynamic_cast<const GeometryModel::Africa<dim>*> 		(&this->get_geometry_model());

    	           //const GeometryModel::BoxAddBI<dim> *geometryc  = dynamic_cast<const GeometryModel::BoxAddBI<dim>*> (this->geometry_model);
    	        Assert (geometryb !=0 || geometryc != 0, // || geometryd != 0, //Assert (geometry != 0 || geometryb !=0 || geometryc != 0,
    	                ExcMessage ("This initial condition can only be used if the geometry "
    	                            "is a box, spherical shell or WSG84."));
    	  Point<dim> InternalPosition;
    	    if(geometryc != 0){ // this is not fully correct for geometryd (WGS84) but hopefully good enough for now. TODO: Make a seperate one for WGS84.
    	  	  /// The cartesian coordinates of the spherical mesh are converted to cartesian coordinates of a rectangular mesh,
    	  	  /// so the rest of the code can stay the same (it was originally build for a rectangula mesh).
    	  ///2d
    	  	  if(dim==2){
    			  const double InnerRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).inner_radius();
    			  const double OuterRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius();
    			  InternalPosition[0]=(atan2((position[dim-1]),position[0])*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
    			  InternalPosition[dim-1]=OuterRadius-this->get_geometry_model().depth(position); //OuterRadius-sqrt(pow(position[0],2)*pow(position[dim-1]+InnerRadius,2));
    			  //Height_to_surface = OuterRadius;


    	  		  //InternalPosition[dim-1]=dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius()-sqrt(pow(position[0],2)*pow(position[dim-1],2));
    	  		   }else{
    	  			 const double InnerRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).inner_radius();
    	  			 const double OuterRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius();
    	  			 InternalPosition[0]=(acos(position[2]/(OuterRadius-this->get_geometry_model().depth(position)))*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
    	  			 InternalPosition[1]=(atan2((position[dim-1]),position[0])*(180/numbers::PI)*numbers::PI*(OuterRadius))/180;
    	  			 InternalPosition[2]=OuterRadius-this->get_geometry_model().depth(position);
    	  	  }

    	  }
		 else{
    		  InternalPosition=position;
    	  }

      double x = InternalPosition[0];
      double z = InternalPosition[dim-1];
      double Slab_dip_rad = 			(Slab_dip/180.0)*numbers::PI;
      double upper_slab_cutoff=			x*tan(-Slab_dip_rad)+Initial_subduction_point*tan(Slab_dip_rad)+Height_to_surface;
      double lower_slab_cutoff=			upper_slab_cutoff-(Crustal_thickness_subducting_plate/sin(Slab_dip_rad)*tan(Slab_dip_rad));
      double lower_crust_rotation_point=	Initial_subduction_point-Crustal_thickness_subducting_plate*tan(0.5*Slab_dip_rad);
      double slab_lower_crust_elevation = 	Height_to_surface-Crustal_thickness_subducting_plate; /// gives the elevation of the lower crustslab 
      double crust_thinning_mantle_angle = 	abs(atan(Crustal_thickness_subducting_plate/(Slab_length-pow(pow(Lithospheric_thickness_overriding_plate,2)+pow(Lithospheric_thickness_overriding_plate/tan(Slab_dip_rad),2),0.5)))); /// gives the angle nessesary for making the crust linear thinner when the slab comes in the mantle
      double crust_thinning_in_mantle = 	x*tan(-Slab_dip_rad-crust_thinning_mantle_angle)+Height_to_surface-Lithospheric_thickness_overriding_plate+(Initial_subduction_point+(Lithospheric_thickness_overriding_plate/tan(Slab_dip_rad)))*tan(Slab_dip_rad+crust_thinning_mantle_angle); /// makes the crust linear thinner when the slab comes in the mantle
      double right_slab_cutoff = 		x*tan(0.5*numbers::PI-Slab_dip_rad)-(Slab_length*sin(Slab_dip_rad)+(Initial_subduction_point+Slab_length*cos(Slab_dip_rad))/tan(Slab_dip_rad)-Height_to_surface); /// cuts the bottom-right part of the slab off.
      double circle_common_part = 		Initial_subduction_point-sin(0.5*Slab_dip_rad)*Crustal_thickness_subducting_plate-0.5*Radius_smoothing_cricle*sin(Slab_dip_rad)/(pow(sin(atan((Radius_smoothing_cricle*sin(Slab_dip_rad))/(Radius_smoothing_cricle-Radius_smoothing_cricle*cos(Slab_dip_rad)))),2));
      
      /// postion[0]=x; InternalPosition[dim-1]=y/z=Height_to_surface
      double composition = 0;
      if(InternalPosition[dim-1]>Height_to_surface&&b_Use_sticky_air==true){ // sticky air
	if(n_comp==3){
	composition = 1;
	}
      }else if((z<=Height_to_surface||b_Use_sticky_air==false)&&(z>=(slab_lower_crust_elevation)&&x<Initial_subduction_point||(z<upper_slab_cutoff&&z>=lower_slab_cutoff&&z<crust_thinning_in_mantle&&x>=lower_crust_rotation_point&&z>=right_slab_cutoff)||(z>=slab_lower_crust_elevation-Radius_smoothing_cricle+Radius_smoothing_cricle*cos(Slab_dip_rad)&&z<slab_lower_crust_elevation&&x>=circle_common_part&&x<circle_common_part+Radius_smoothing_cricle*sin(Slab_dip_rad)&&(pow(x-(circle_common_part),2)+pow((z-(slab_lower_crust_elevation-Radius_smoothing_cricle)),2)>=pow(Radius_smoothing_cricle,2))&&(z<upper_slab_cutoff)))){ // slab plate crust
	if(n_comp==1){ // Oceanic crust
	composition = 1;
	}
      }else if(z>=upper_slab_cutoff&&z>=Height_to_surface-Crustal_thickness_overriding_plate){ // overriding plate crust
	if(n_comp==2){ // Continental crust
	composition = 1;
	}
////////////////////////////////////////////////////////////////////////////////////////////////////
      }else
      {
	double phase_trans_func0 = 0.5*(1.0 + tanh(((Height_to_surface-position[1]) - transition_depths[0] - Clapeyron_slopes[0] * (transition_depths[0] / transition_pressures[0]) * (pTsm*exp(thermal_alpha*9.81*(Height_to_surface-position[1])/reference_specific_heat) - pTsm*exp(thermal_alpha*9.81*transition_depths[0]/reference_specific_heat))) / transition_widths[0]));
	double phase_trans_func1 = 0.5*(1.0 + tanh(((Height_to_surface-position[1]) - transition_depths[1] - Clapeyron_slopes[1] * (transition_depths[1] / transition_pressures[1]) * (pTsm*exp(thermal_alpha*9.81*(Height_to_surface-position[1])/reference_specific_heat) - pTsm*exp(thermal_alpha*9.81*transition_depths[1]/reference_specific_heat))) / transition_widths[1]));

	if(n_compositional_fields == 7){ // If phase transitions, than:
	  if(n_comp==3){ // Transition zone 
	    composition = phase_trans_func0 - phase_trans_func1;
	  }

	  if(n_comp==4){ // Lower mantle 
	    composition = phase_trans_func1;
	  }
	  if(n_comp==0){ // What is left, is the upper mantle
	    composition = 1.0 - phase_trans_func0;
	  }}
	else{ // If no phase transitions, than:
	  if(n_comp==0){ // What is left, is the upper mantle
	    composition = 1.0;
	  }}
      }
////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////// introduce weakzones /////////////////////////////////////////////////
     for(unsigned int ii=0;ii<wkz_n_zones;++ii){
    if(InternalPosition[dim-1]<wkz_upperboundary[ii]&&InternalPosition[dim-1]>wkz_lowerboundary[ii]&&wkz_composition[ii]>=0){
	 double wkz_composition_abs=abs(wkz_composition[ii]);
       double highest_weakening = 0;
 	    //// set pmin and pmax to the bounderies of the field which needs to be refined ////
	    std::vector<double> pmin;
	    std::vector<double> pmax;
	    pmin.resize(dim,0);
	    pmax.resize(dim,0);      
	    Point<dim> used_rotationpoint;
	    //// determine what rotation point should be used //// //todo make 3d
	    if(wkz_rotationpoint[ii]==1){
	      pmin[0]=wkz_beginpoints[ii][0];
	      pmax[0]=wkz_beginpoints[ii][0]+wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+0.5*wkz_thickness[ii];
	      ///used_rotationpoint[0]=wkz_beginpoints[ii][0];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];
	    }else if(wkz_rotationpoint[ii]==2){
	      pmin[0]=wkz_beginpoints[ii][0];
	      pmax[0]=wkz_beginpoints[ii][0]+wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	    }else if(wkz_rotationpoint[ii]==3){
	      pmin[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0]+0.5*wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	    }else if(wkz_rotationpoint[ii]==4){
	      pmin[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-wkz_thickness[ii];
	    }else if(wkz_rotationpoint[ii]==5){
	      pmin[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+0.5*wkz_thickness[ii];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];      
	    }else if(wkz_rotationpoint[ii]==6){
	      pmin[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+wkz_thickness[ii];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-wkz_length[ii];
	     // used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1];
	    }else if(wkz_rotationpoint[ii]==7){
	      pmin[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0]+0.5*wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+wkz_thickness[ii];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1];
	    }else if(wkz_rotationpoint[ii]==8){
	      pmin[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      pmax[0]=wkz_beginpoints[ii][0]+0.5*wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+0.5*wkz_thickness[ii];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0]-0.5*wkz_length[ii];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1]-0.5*wkz_thickness[ii];    
	    }else{
	      pmin[0]=wkz_beginpoints[ii][0];
	      pmax[0]=wkz_beginpoints[ii][0]+wkz_length[ii];
	      pmin[dim-1]=wkz_beginpoints[ii][dim-1];
	      pmax[dim-1]=wkz_beginpoints[ii][dim-1]+wkz_thickness[ii];
	      //used_rotationpoint[0]=wkz_beginpoints[ii][0];
	      //used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1];
	    }
	    if(dim==3){
	    pmin[dim]=wkz_beginpoints[ii][dim];
	    pmax[dim]=wkz_beginpoints[ii][dim];
	    }
	    //pmin[0]=wkz_beginpoints[ii][0];
	    //pmax[0]=wkz_beginpoints[ii][0]+wkz_length[ii];
	    //pmin[1]=wkz_beginpoints[ii][dim-1];
	    //pmax[1]=wkz_beginpoints[ii][dim-1]+wkz_thickness[ii];
	    used_rotationpoint[0]=wkz_beginpoints[ii][0];
	    used_rotationpoint[dim-1]=wkz_beginpoints[ii][dim-1];
       
        ////// rotation of current coordinate ///////
	    Point<dim> non_rotated_position = InternalPosition;
	    double rad_angle=(wkz_angle[ii]/180.0)*numbers::PI;
	    Point<dim> rotated_position= InternalPosition;
	    for(unsigned int i=0; i<dim; ++i){
	    non_rotated_position[i]=non_rotated_position[i]-used_rotationpoint[i];
	    }
	    rotated_position[0]=non_rotated_position[0] * cos(rad_angle) - non_rotated_position[dim-1]*sin(rad_angle);
	    rotated_position[dim-1]=non_rotated_position[0] * sin(rad_angle) + non_rotated_position[dim-1]*cos(rad_angle);
	    for(unsigned int i=0; i<dim; ++i){
	    rotated_position[i]=rotated_position[i]+used_rotationpoint[i];
	    }
	 //// end rotation of current coordinate ////
	 //std::cout << "flag 1!" << std::endl;
	if(wkz_feature[ii]=="ellipse"||wkz_feature[ii]=="cut corners"){
	 //std::cout << "flag ellipse!" << std::endl;
	  double a = 0.5*(pmax[0]-pmin[0]);
	  double b = 0.5*(pmax[dim-1]-pmin[dim-1]);
	  std::vector<double> c(3,0);
	  c[0]=pmin[0]+a;
	  c[1]=pmin[dim-1]+b;
	  
	  if(dim==2&&((pow(rotated_position[0]-c[0],2)/pow(a,2))+(pow(rotated_position[dim-1]-c[dim-1],2)/pow(b,2))<1)){
	if(n_comp==wkz_composition_abs){
	 composition=1;
	 //std::cout << "flag 3!" << std::endl;
	}else{
	  composition=0;
	}
	  }
	}else if(wkz_feature[ii]=="triangle up"){
	  double gradient = (pmax[dim-1]-pmin[dim-1])/(0.5*(pmax[0]-pmin[0]));
	  double c = pmin[0]*gradient;
	  if(rotated_position[dim-1]>pmin[dim-1]&&rotated_position[dim-1]<pmin[dim-1]-c+rotated_position[0]*gradient&&rotated_position[dim-1]<pmax[dim-1]+(pmax[dim-1]+c-pmin[dim-1])-rotated_position[0]*gradient){
	    
	    if(n_comp==wkz_composition_abs){
	 composition=1;
	}else{
	  composition=0;
	}   
           
	  }
	}else if(wkz_feature[ii]=="triangle down"){
	  double gradient = (pmax[dim-1]-pmin[dim-1])/(0.5*(pmax[0]-pmin[0]));
	  double c = pmin[0]*gradient;
	  if(rotated_position[dim-1]<pmax[dim-1]&&rotated_position[dim-1]>pmin[dim-1]-c-(pmax[dim-1]-pmin[dim-1])+rotated_position[0]*gradient&&rotated_position[dim-1]>pmax[dim-1]+(pmax[dim-1]+c-(pmax[dim-1]-pmin[dim-1])-pmin[dim-1])-rotated_position[0]*gradient){
	
       if(n_comp==wkz_composition_abs){
	 composition=1;
	}else{
	  composition=0;
	}
       
	  } 
	}else if(wkz_feature[ii]=="triangle right"){
	  double gradient = (0.5*(pmax[dim-1]-pmin[dim-1]))/(pmax[0]-pmin[0]);
	  double c = pmin[0]*gradient;
	  if(rotated_position[0]>pmin[0]&&rotated_position[dim-1]>pmin[dim-1]-c+rotated_position[0]*gradient&&rotated_position[dim-1]<pmax[dim-1]+c-rotated_position[0]*gradient){
	 
       if(n_comp==wkz_composition_abs){
	 composition=1;
	}else{
	  composition=0;
	}
       
	  }
	}else if(wkz_feature[ii]=="triangle left"){
	  double gradient = (0.5*(pmax[dim-1]-pmin[dim-1]))/(pmax[0]-pmin[0]);
	  double c = pmin[0]*gradient;
	  if(rotated_position[0]<pmax[0]&&rotated_position[dim-1]<pmin[dim-1]-c+0.5*(pmax[dim-1]-pmin[dim-1])+rotated_position[0]*gradient&&rotated_position[dim-1]>pmax[dim-1]+c-0.5*(pmax[dim-1]-pmin[dim-1])-rotated_position[0]*gradient){
	
       if(n_comp==wkz_composition_abs){
	 composition=1;
	}else{
	  composition=0;
	}
       
	  }
	}else { /// rectangle
     if(dim==2&&(rotated_position[dim-1]<pmax[dim-1]&&rotated_position[dim-1]>pmin[dim-1])&&(rotated_position[0]<pmax[0]&&rotated_position[0]>pmin[0])){
	 
       if(n_comp==wkz_composition_abs){
	 composition=1;
	}else{
	  composition=0;
	}
       
     }
     }
       
     }
    }
   ////////////////////////////////////////////// end introduction weakzones //////////////////////////////////////////////
   
      return composition;
    }

    template <int dim>
    void
    Subduction4<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Compositional initial conditions");
      {
        prm.enter_subsection("Subduction");
        {
	  prm.declare_entry ("Use sticky air", "true",
                Patterns::Bool (),
                "Let model use sticky air or not.");

        	prm.declare_entry ("Subducting crust layers", "1",
        	                             Patterns::Integer (0),
        	                             "The amount of subducting crust layers");
        	prm.declare_entry ("Overriding crust layers", "1",
        	                             Patterns::Integer (0),
        	                             "The amount of overriding crust layers");
          prm.declare_entry ("Height to surface", "660e3", /// Height_to_surface -> Height_to_surface
                             Patterns::Double (0),
                             "The distance from the bottom of the model the the surface in meters. Above it there will be a sticky air/water layer.");
	  prm.declare_entry ("Initial subduction point", "500e3", /// Initial_subduction_point -> Initial_subduction_point
                             Patterns::Double (0),
                             "The x position where the slab goes down in the mantle in meters");
          prm.declare_entry ("Slab dip", "30", /// dip -> Slab_dip
                             Patterns::Double (),
                             "The intial dip of the subducting slab in degrees.");
	  prm.declare_entry ("Slab length", "300e3", /// d -> Slab_length
                             Patterns::Double (0),
                             "The length the slab penetrates into the mantle from the surface measured along the slab in meters.");
          prm.declare_entry ("Crustal thickness subducting plate", "5e3", ///Crustal_thickness_subducting_plate -> Crustal_thickness_subducting_plate
                             Patterns::Double (0),
                             "The thickness of the crust of the subduction plate in meters.");
          /*prm.declare_entry ("Lithospheric thickness subducting plate", "80e3", /// l -> Lithospheric_thickness_subducting_plate
                             Patterns::Double (0),
                             "The  complete thickness of the subducting plate in meters. Check if it can be removed."
                             "In meters.");*/
          prm.declare_entry ("Crustal thickness overriding plate", "5e3", ///lcr3 -> Crustal_thickness_overriding_plate
                             Patterns::Double (0),
                             "The thickness of the crust of the overring plate in meters.");
          prm.declare_entry ("Lithospheric thickness overriding plate", "100e3", /// Lithospheric_thickness_overriding_plate-> Lithospheric_thickness_overriding_plate
                             Patterns::Double (0),
                             "The thickness of the complete lithosphere in meters. Check if it can be removed");
          prm.declare_entry ("Radius smoothing cricle", "7.5e4", /// r -> Radius_smoothing_cricle
                             Patterns::Double (0),
                             "The bottom part of the tranisition in the subducting plate from normal to dipping can be smoothened by a circle (in meters).");
          /*prm.declare_entry ("Weak zone thickness", "30e3", /// dwz (remove, also remove lwzcr)
                             Patterns::Double (0),
                             "remove (and make weak zones available).");*/
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      prm.enter_subsection("Material model");
       {
////////////////////////////////////////////////////////////////////////////////////////////////////
        prm.enter_subsection("Viscoplastic model");
        {
          prm.enter_subsection("Phase transitions");
          {
            prm.declare_entry ("List of transition depths", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
            prm.declare_entry ("List of Clapeyron slopes", "",
                           Patterns::List (Patterns::Double()),
                           "");
            prm.declare_entry ("List of transition widths", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
            prm.declare_entry ("List of transition pressures", "",
                           Patterns::List (Patterns::Double(0)),
                           "");
	  }
	  prm.leave_subsection();

	  prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the reference specific heat $cp$. "
                             "Units: $J/kg/K$.");
	  prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
	}
	prm.leave_subsection();

	prm.enter_subsection("Initial conditions");
      	{
          prm.enter_subsection("Subduction temperature");
          {
	    prm.declare_entry ("Potential temperature mantle on surface", "1600.",
                               Patterns::Double (0),
                               "");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
////////////////////////////////////////////////////////////////////////////////////////////////////

        prm.enter_subsection("Stokesloop model");
        {
	  prm.enter_subsection("Weakzones");
	  { 
	     prm.declare_entry ("Number of weakzones", "0",
                          Patterns::Integer (0),
                          "The number of weakzones that in the model");
	     prm.declare_entry ("List of existance times", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of how long the weakzone should exist in seconds.");
	     prm.declare_entry ("List of existance steps", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of how long the weakzone should exist in timesteps.");
	     prm.declare_entry ("List of weakening", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of how much the viscosity in the section should devided by.");
	     prm.declare_entry ("List of order functions x-axis", "",
                          Patterns::List (Patterns::Double()),
                          "A list of how the weakening should decrease from the middel of the section to the borders on the x-axis.");
	     prm.declare_entry ("List of order functions y-axis", "",
                          Patterns::List (Patterns::Double()),
                          "A list of how the weakening should decrease from the middel of the section to the borders on the y-axis.");
	     prm.declare_entry ("List of rotationpoints", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of where the refinment box should be rotated from: 0=lowerleft, 1=middleleft, 2=upperleft, 3=uppermiddle,..., 8=center, all values will default to 0.");
	     prm.declare_entry ("List of beginpoints", "",
                          Patterns::List(Patterns::List(Patterns::Double(0),0,100000000,":"),0,100000000,","),
                          "A list of points where the weakzone should begin.");
	     prm.declare_entry ("List length of weakzones", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of how long the weakzone should be.");
	     prm.declare_entry ("List of angle of weakzone", "",
                          Patterns::List (Patterns::Double()),
                          "A list of what angle the weakzone should go.");
	     prm.declare_entry ("List of thickness of weakzone", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of what thickness the weakzone should go.");
	     prm.declare_entry ("List of features of weakzones", "",
                          Patterns::List (Patterns::Anything ()),
                          "A list of what feature should be used.");
	     prm.declare_entry ("List of refinment errors", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of how much error should be added to the refinment error. Note that the weakzone refiment module must be enabled.");
	     prm.declare_entry ("List of add to initial temperature", "",
                          Patterns::List (Patterns::Double()),
                          "A list of how many degrees Kelvin should be added to the temperature field. Note that a compatible intitial temperature module must be enabled.");
	     prm.declare_entry ("List of upper boundary", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of at what upper boundary the effect should be cut off.");
	     prm.declare_entry ("List of lower boundary", "",
                          Patterns::List (Patterns::Double(0)),
                          "A list of at what lower boundary the effect should be cut off.");
	   }        
	prm.leave_subsection();
	}
	prm.leave_subsection();
       }
       prm.leave_subsection();
    }


    template <int dim>
    void
    Subduction4<dim>::parse_parameters (ParameterHandler &prm)
    {
      // we need to get at the number of compositional fields here to
      // initialize the function parser. unfortunately, we can't get it
      // via SimulatorAccess from the simulator itself because at the
      // current point the SimulatorAccess hasn't been initialized
      // yet. so get it from the parameter file directly.
      prm.enter_subsection ("Compositional fields");
//      const unsigned int n_compositional_fields = prm.get_integer ("Number of fields");
        n_compositional_fields = prm.get_integer ("Number of fields");
      prm.leave_subsection ();

      prm.enter_subsection("Compositional initial conditions");
      {
        prm.enter_subsection("Subduction");
        {
	 b_Use_sticky_air = 				prm.get_bool   ("Use sticky air");
         i_Subducting_crust_layers = 			prm.get_integer("Subducting crust layers");
         i_Overrding_crust_layers = 			prm.get_integer("Overriding crust layers");
         Height_to_surface = 				prm.get_double ("Height to surface"); /// Height_to_surface
         Initial_subduction_point = 			prm.get_double ("Initial subduction point"); /// Initial_subduction_point
         Slab_dip = 					prm.get_double ("Slab dip"); /// dip
         Slab_length = 					prm.get_double ("Slab length"); /// d
         Crustal_thickness_subducting_plate = 		prm.get_double ("Crustal thickness subducting plate"); /// Crustal_thickness_subducting_plate
         //Lithospheric_thickness_subducting_plate = 	prm.get_double ("Lithospheric thickness subducting plate"); /// l
         Crustal_thickness_overriding_plate = 		prm.get_double ("Crustal thickness overriding plate"); /// lcr3
         Lithospheric_thickness_overriding_plate = 	prm.get_double ("Lithospheric thickness overriding plate"); /// Lithospheric_thickness_overriding_plate
	 Radius_smoothing_cricle =		 	prm.get_double ("Radius smoothing cricle"); /// r
	 
	  //function.reset (new Functions::ParsedFunction<dim>(n_compositional_fields));
          //function->parse_parameters (prm);
	  //std::cout << "ending compositional ic; Dim: " << dim << std::endl;
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Material model");
       {
////////////////////////////////////////////////////////////////////////////////////////////////////
        prm.enter_subsection("Viscoplastic model");
        {
          prm.enter_subsection("Phase transitions");
          {
            transition_depths             = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition depths")));

            Clapeyron_slopes              = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of Clapeyron slopes")));

            transition_widths             = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition widths")));

            transition_pressures          = Utilities::string_to_double
                                          (Utilities::split_string_list(prm.get ("List of transition pressures")));
	  }
	  prm.leave_subsection();
	
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
	}
	prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Subduction temperature");
        {
          pTsm                  = prm.get_double ("Potential temperature mantle on surface");
        }
        prm.leave_subsection();
      } 
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
////////////////////////////////////////////////////////////////////////////////////////////////////

        prm.enter_subsection("Stokesloop model");
        {
      /// do stuff for weakzones
          prm.enter_subsection("Weakzones");
          {
	    wkz_n_zones = prm.get_integer ("Number of weakzones");

            if (wkz_n_zones > 0){
          
          //parametes needed for all weakzones
 
          const std::vector<double> n_wkz_orderx = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of order functions x-axis")));
          wkz_orderx = std::vector<double> (n_wkz_orderx.begin(),
                                                         n_wkz_orderx.end());
          AssertThrow (wkz_orderx.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of orders of decrease"));
	  
	  
          const std::vector<double> n_wkz_ordery = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of order functions y-axis")));
          wkz_ordery = std::vector<double> (n_wkz_ordery.begin(),
                                                         n_wkz_ordery.end());
          AssertThrow (wkz_ordery.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of orders of decrease"));
	  
	  
          const std::vector<double> n_wkz_rotationpoint = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of rotationpoints")));
          wkz_rotationpoint = std::vector<double> (n_wkz_rotationpoint.begin(),
                                                         n_wkz_rotationpoint.end());
          AssertThrow (wkz_rotationpoint.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of rotationpoints"));
          
	  
          std::vector<std::string> n_wkz_beginpoints_outer = Utilities::split_string_list(prm.get ("List of beginpoints"));
	  int size_element = n_wkz_beginpoints_outer.size();
	  wkz_beginpoints.resize(size_element,std::vector<double>(dim,0));
	  for ( unsigned int it1 = 0; it1 != n_wkz_beginpoints_outer.size(); ++it1 )
	  {
	    std::vector<double> n_wkz_beginpoints_inner = Utilities::string_to_double(Utilities::split_string_list(n_wkz_beginpoints_outer[it1],':'));
	  if(dim==2){
	    std::vector<double> tmpPoint(dim,0);
            tmpPoint[0] = n_wkz_beginpoints_inner[0];
	    tmpPoint[1] = n_wkz_beginpoints_inner[1];
            wkz_beginpoints[it1] = tmpPoint;
	    //std::cout << "end: " << wkz_beginpoints[0][0] << "," << wkz_beginpoints[0][1] << "; " << wkz_beginpoints[1][0] << "," << wkz_beginpoints[1][1] << "; " << wkz_beginpoints[2][0] << "," << wkz_beginpoints[2][1] << std::endl;
	  }else if(dim==3){
	    std::vector<double> tmpPoint(dim,0);
            tmpPoint[0] = n_wkz_beginpoints_inner[0];
	    tmpPoint[1] = n_wkz_beginpoints_inner[1];
	    tmpPoint[2] = n_wkz_beginpoints_inner[2];
            wkz_beginpoints[it1] = tmpPoint;
	    //wkz_beginpoints[itteration_number] = Point<dim> (n_wkz_beginpoints_inner[0],
                                                        // n_wkz_beginpoints_inner[1],
							// n_wkz_beginpoints_inner[2]);  
	  }else{
	    AssertThrow (true,ExcMessage("Problem with dimentions (not 2 and not 3)."));
	  }
          AssertThrow (n_wkz_beginpoints_inner.size() == dim,
                       ExcMessage("Invalid input parameter file: Wrong number of entries for a coordinate in List of beginpoints"));
	  }
          AssertThrow (wkz_beginpoints.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of beginpoints"));
	  
	  
          
          const std::vector<double> n_wkz_length = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List length of weakzones")));
          wkz_length = std::vector<double> (n_wkz_length.begin(),
                                                         n_wkz_length.end());
          AssertThrow (wkz_length.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List length of weakzones"));
	  
	  
          
          const std::vector<double> n_wkz_angle = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of angle of weakzone")));
          wkz_angle = std::vector<double> (n_wkz_angle.begin(),
                                                         n_wkz_angle.end());
          AssertThrow (wkz_angle.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of angle of weakzone"));
	  
	  
          
          const std::vector<double> n_wkz_thickness = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of thickness of weakzone")));
          wkz_thickness = std::vector<double> (n_wkz_thickness.begin(),
                                                         n_wkz_thickness.end());
          AssertThrow (wkz_thickness.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of thickness of weakzone"));
	  
	  
          
          const std::vector<std::string> n_wkz_feature = Utilities::split_string_list(prm.get ("List of features of weakzones"));
          wkz_feature = std::vector<std::string> (n_wkz_feature.begin(),
                                                         n_wkz_feature.end());
          AssertThrow (wkz_feature.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of features of weakzones"));
	  
	  
          
          const std::vector<double> n_wkz_upperboundary = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of upper boundary")));
          wkz_upperboundary = std::vector<double> (n_wkz_upperboundary.begin(),
                                                         n_wkz_upperboundary.end());
          AssertThrow (wkz_upperboundary.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of upper boundary"));
	  
	  
          
          const std::vector<double> n_wkz_lowerboundary = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of lower boundary")));
          wkz_lowerboundary = std::vector<double> (n_wkz_lowerboundary.begin(),
                                                         n_wkz_lowerboundary.end());
          AssertThrow (wkz_lowerboundary.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of lower boundary"));
	  
	  }
      }
      prm.leave_subsection (); 
      /// end do stuff for weakzones
      
	}
	prm.leave_subsection();
       }
       prm.leave_subsection();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace CompositionalInitialConditions
  {
    ASPECT_REGISTER_COMPOSITIONAL_INITIAL_CONDITIONS(Subduction4,
                                                     "subduction4",
                                                     "Composition is given in for a subduction setting")
  }
}

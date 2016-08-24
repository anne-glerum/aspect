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
/*  $Id: mckenzie.cc 1088 2013-10-22 13:45 glerum $  */ 


#include <aspect/initial_conditions/subduction_temp2.h>
#include <aspect/geometry_model/box.h>
//#include <aspect/geometry_model/boxaddbi.h>
//#include <aspect/geometry_model/asymbox.h>
#include <aspect/geometry_model/spherical_shell.h>
//#include <aspect/geometry_model/wgs84.h>
//#include <aspect/geometry_model/africa.h>

#include <cmath>

#include <iostream>
#include <fstream>

namespace aspect
{
  namespace InitialConditions
  {
    template <int dim>
    double
    SubductionTemp2<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // This initial condition only makes sense if the geometry is a
      // box. verify that it is indeed a box
//        const GeometryModel::AsymBox<dim> 			*geometry = 	dynamic_cast<const GeometryModel::AsymBox<dim>*> 		(&this->get_geometry_model());
        const GeometryModel::Box<dim> 				*geometryb = 	dynamic_cast<const GeometryModel::Box<dim>*> 			(&this->get_geometry_model());
        const GeometryModel::SphericalShell<dim> 	*geometryc = 	dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model());
  //      const GeometryModel::WGS84<dim> 			*geometryd = 	dynamic_cast<const GeometryModel::WGS84<dim>*> 			(&this->get_geometry_model());
        //const GeometryModel::Africa<dim> 			*geometrye = 	dynamic_cast<const GeometryModel::Africa<dim>*> 		(&this->get_geometry_model());

        //const GeometryModel::BoxAddBI<dim> *geometryc  = dynamic_cast<const GeometryModel::BoxAddBI<dim>*> (this->geometry_model);
      Assert (geometryb !=0 || geometryc != 0, //|| geometryd != 0, //Assert (geometry != 0 || geometryb !=0 || geometryc != 0,
              ExcMessage ("This initial condition can only be used if the geometry "
                          "is a box, spherical shell, africa or WSG84."));
Point<dim> InternalPosition;
  if(geometryc != 0){ // this is not fully correct for geometryd (WGS84) but hopefully good enough for now. TODO: Make a seperate one for WGS84.
	  /// The cartesian coordinates of the spherical mesh are converted to cartesian coordinates of a rectangular mesh,
	  /// so the rest of the code can stay the same (it was originally build for a rectangula mesh).
///2d
	  if(dim==2){
		  const double InnerRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).inner_radius();
		  const double OuterRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius();
		  InternalPosition[0]=(atan2((position[1]),position[0])*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
		  InternalPosition[1]=OuterRadius-this->get_geometry_model().depth(position); //OuterRadius-sqrt(pow(position[0],2)*pow(position[dim-1]+InnerRadius,2));
		  //depth_internal = OuterRadius;
		    //InternalPosition[dim-1]=dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius()-sqrt(pow(position[0],2)*pow(position[dim-1],2));
		  	  }else{
		  		const double InnerRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).inner_radius();
		  		const double OuterRadius = dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius();
		  		InternalPosition[0]=(acos(position[2]/(OuterRadius-this->get_geometry_model().depth(position)))*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
		  		InternalPosition[1]=(atan2((position[dim-1]),position[0])*(180/numbers::PI)*numbers::PI*(OuterRadius))/180;
		  		InternalPosition[2]=OuterRadius-this->get_geometry_model().depth(position);
	  }

}/*else if(geometryd != 0){
	  ///2d
	     if(dim==2){
	      	const double InnerRadius = 5711000; //dynamic_cast<const GeometryModel::WGS84<dim>&>(this->get_geometry_model()).inner_radius();
	      	const double OuterRadius = 6371000; //dynamic_cast<const GeometryModel::WGS84<dim>&>(this->get_geometry_model()).outer_radius();
	      	InternalPosition[0]=(atan2((position[dim-1]),position[0])*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
	      	InternalPosition[dim-1]=OuterRadius-this->get_geometry_model().depth(position); //OuterRadius-sqrt(pow(position[0],2)*pow(position[dim-1]+InnerRadius,2));
	      	//Height_to_surface = OuterRadius;


	      	//InternalPosition[dim-1]=dynamic_cast<const GeometryModel::SphericalShell<dim>&>(this->get_geometry_model()).outer_radius()-sqrt(pow(position[0],2)*pow(position[dim-1],2));
	      }else{
	      	const double InnerRadius = 5711000; //dynamic_cast<const GeometryModel::WGS84<dim>&>(this->get_geometry_model()).inner_radius();
	      	const double OuterRadius = 6371000; //dynamic_cast<const GeometryModel::WGS84<dim>&>(this->get_geometry_model()).outer_radius();
	      	InternalPosition[0]=(acos(position[2]/(OuterRadius-this->get_geometry_model().depth(position)))*(180/numbers::PI)*2*numbers::PI*(OuterRadius))/360;
	      	InternalPosition[1]=(atan2((position[dim-1]),position[0])*(180/numbers::PI)*numbers::PI*(OuterRadius))/180;
	      	InternalPosition[2]=OuterRadius-this->get_geometry_model().depth(position);
	      }
}*/ else{
InternalPosition=position;
}
  double g=dynamic_cast<const GravityModel::Interface<dim>&>(this->get_gravity_model()).gravity_vector(position).norm();
//NB:
//I replaced 1553 by T_pot
//I replaced 273 by C_K_conv
//kappa is k
//Please comment out the following when the input file is set to correct values for density, Cp, k and alfa
//     const double density = 3300.0;
//     const double Cp = 1250.0;
//     const double k = 4.0;
//     const double alfa = 0.00003;
////////////   

     //const double C_K_conv = 273.0; 
     const double dip_rad = (dip/180.0)*numbers::PI;
     const double R = (density*Cp*(v/(365.25*24.0*60.0*60.0))*l)/(2.0*k);
     const double H = (Cp/(alfa*g*l));
//     std::cout << "H:" << H << ";R:" << R << std::endl;
     const double position_x = (((InternalPosition[0]-(Px-l*sin(dip_rad)))*cos((dip/180.0)*numbers::PI)-
                               (InternalPosition[dim-1]-(depth-l*cos(dip_rad)))*sin((dip/180.0)*numbers::PI))/l);
     const double position_z = (((InternalPosition[0]-(Px-l*sin(dip_rad)))*sin((dip/180.0)*numbers::PI)+
                               (InternalPosition[dim-1]-(depth-l*cos(dip_rad)))*cos((dip/180.0)*numbers::PI))/l);
     
     double temp;
   //sticky air
   if(InternalPosition[dim-1]>depth)
   {
     temp=Ttop;
     //std::cout << Ttop << std::endl;
   }
   //c2plate
   else if(InternalPosition[dim-1]>=(depth-l)&&
      InternalPosition[dim-1]>=(InternalPosition[0]*tan(0.5*numbers::PI-((0.5*dip)/180.0)*numbers::PI)-(d*sin(0.5*dip_rad)+
                       (Px+d*cos(0.5*dip_rad))/tan(0.5*dip_rad)-depth-(d/sin(0.5*dip_rad)))))
     {
      temp = (depth-InternalPosition[dim-1])*((pTsm-(((pTsm*alfa*g)/Cp)*1000.0)*(l/1000.0)-Ttop)/l)+Ttop;
     }

   //c2slab
   else if(InternalPosition[dim-1]<(InternalPosition[0]*tan(-dip_rad)+Px*tan(dip_rad)+depth)&&
	   InternalPosition[dim-1]<(InternalPosition[0]*tan(-dip_rad-abs(atan(lcr2/(d-pow(pow(lc3,2)+pow(lc3/tan(dip_rad),2),0.5)))))+depth-lc3+(Px+(lc3/tan(dip_rad)))*tan(dip_rad+abs(atan(lcr2/(d-pow(pow(lc3,2)+pow(lc3/tan(dip_rad),2),0.5))))))&&
           InternalPosition[dim-1]>=(InternalPosition[0]*tan(-dip_rad)+Px*tan(dip_rad)+depth-(l/sin(dip_rad)*tan(dip_rad)))&&
           InternalPosition[dim-1]<(InternalPosition[0]*tan(0.5*numbers::PI-dip_rad)-(d*sin(dip_rad)+(Px+d*cos(dip_rad))/tan(dip_rad)
                            -depth-(d/sin(dip_rad))))&&
           InternalPosition[dim-1]>=(InternalPosition[0]*tan(0.5*numbers::PI-dip_rad)-(d*sin(dip_rad)+(Px+d*cos(dip_rad))/tan(dip_rad)-depth)))
          {
           temp = exp((position_x*sin(dip_rad)-position_z*cos(dip_rad))/H);
           double sum=0;
           for (int i=1;i<=n_sum;i++)
               {
                sum += (std::pow((-1.0),i)/(i*numbers::PI)) * 
                      (exp((R-std::pow(std::pow(R,2.0)+std::pow(i,2.0)*std::pow(numbers::PI,2.0),0.5))*position_x)) * 
                      (sin(i*numbers::PI*position_z));
		      //std::cout << "sum" << i << ":" << sum;
               }
           //std::cout << "\tT:" << temp;
           temp = temp * (T_pot+2.0*(T_pot-Ttop) * sum);
	   //std::cout << "\tc=" << InternalPosition[0] << ":" << InternalPosition[dim-1] << "\tcp=" << position_x << ":" << position_z << ";\tsum:" << sum << ";\tT:" << temp << std::endl;
          }

   //c3plate
   else if (InternalPosition[dim-1]>=(InternalPosition[0]*tan(-dip_rad)+Px*tan(dip_rad)+depth+dwz/cos(dip_rad))&&
            InternalPosition[dim-1]>=depth-lc3)
           { 
            temp = (depth-InternalPosition[dim-1])*((pTsm-(((pTsm*alfa*g)/Cp)*1000.0)*(lc3/1000.0)-Ttop)/lc3)+Ttop;
           }

   //c4wkz
   else if(InternalPosition[dim-1]>=InternalPosition[0]*tan(-dip_rad)+Px*tan(dip_rad)+depth&&
           InternalPosition[dim-1]<InternalPosition[0]*tan(-dip_rad)+Px*tan(dip_rad)+depth+dwz/cos(dip_rad)&&
           InternalPosition[dim-1]>=depth-lc3)
          {
           temp = (depth-InternalPosition[dim-1])*((pTsm-(((pTsm*alfa*g)/Cp)*1000.0)*(lc3/1000.0)-Ttop)/lc3)+Ttop;
          }

   //c1mantle
   else 
       {

////////////////////////////////////////////////////////////////////////////////////////////////////

	  temp = pTsm*exp(alfa*g*(depth-InternalPosition[dim-1])/Cp);

//          double tempOld = pTsm+(((pTsm*alfa*g)/Cp)*1000.0)*((depth-InternalPosition[dim-1])/1000.0);

//	if (this->get_timestep_number() == 0){
//	  if(position[0] > 0 && position[0] < 15.e3){
//   	    std::ofstream myfile;
//    	    myfile.open ("/home/cblom/ASPECT/aspect10juni215/aspect/build/subductionmodelM/example.txt");
//    	    myfile << position[0] << position[1] << " " << temp << " " << tempOld  << "\n" << std::flush;
//	    cout << position[0] << " " << position[1] << " " << temp << " " << tempOld << "\n" << std::flush;
//	  }
//	}

////////////////////////////////////////////////////////////////////////////////////////////////////


        //std::cout << "ptsm:" << pTsm << "; alfa:" << alfa << "; g:" << g << "; Cp:" << Cp << "; depth:" << depth << "; intpos:" << InternalPosition[dim-1] << std::endl;
       }
   //debug!
   //temp = InternalPosition[2];
    ///////////////////////////////////////////////// introduce weakzones /////////////////////////////////////////////////

     for(unsigned int ii=0;ii<wkz_n_zones;++ii){
    if(InternalPosition[dim-1]<wkz_upperboundary[ii]&&InternalPosition[dim-1]>wkz_lowerboundary[ii]){
	 
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
	if(wkz_feature[ii]=="ellipse"||wkz_feature[ii]=="cut corners"){
	  double a = 0.5*(pmax[0]-pmin[0]);
	  double b = 0.5*(pmax[dim-1]-pmin[dim-1]);
	  std::vector<double> c(3,0);
	  c[0]=pmin[0]+a;
	  c[1]=pmin[dim-1]+b;
	  
	  if(dim==2&&((pow(rotated_position[0]-c[0],2)/pow(a,2))+(pow(rotated_position[dim-1]-c[dim-1],2)/pow(b,2))<1)){
	 /*
	 double tmptemperature = temp;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmptemperature =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(temp/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  

	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;

	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);
       highest_weakening = wkz_weakening[ii];
       */
	 temp+=wkz_heating[ii];
	  }
	}else if(wkz_feature[ii]=="triangle up"){
	  double gradient = (pmax[dim-1]-pmin[dim-1])/(0.5*(pmax[0]-pmin[0]));
	  double c = pmin[0]*gradient;
	  if(rotated_position[dim-1]>pmin[dim-1]&&rotated_position[dim-1]<pmin[dim-1]-c+rotated_position[0]*gradient&&rotated_position[dim-1]<pmax[dim-1]+(pmax[dim-1]+c-pmin[dim-1])-rotated_position[0]*gradient){
	    /*
	 double tmpvisosity = viscosity;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmpvisosity =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(viscosity/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;

	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);
       highest_weakening = wkz_weakening[ii];*/
       temp+=wkz_heating[ii];
	  }
	}else if(wkz_feature[ii]=="triangle down"){
	  double gradient = (pmax[dim-1]-pmin[dim-1])/(0.5*(pmax[0]-pmin[0]));
	  double c = pmin[0]*gradient;
	  if(rotated_position[dim-1]<pmax[dim-1]&&rotated_position[dim-1]>pmin[dim-1]-c-(pmax[dim-1]-pmin[dim-1])+rotated_position[0]*gradient&&rotated_position[dim-1]>pmax[dim-1]+(pmax[dim-1]+c-(pmax[dim-1]-pmin[dim-1])-pmin[dim-1])-rotated_position[0]*gradient){
	 /*
	 double tmpvisosity = viscosity;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmpvisosity =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(viscosity/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;

	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);
       highest_weakening = wkz_weakening[ii];*/
       temp+=wkz_heating[ii];
       
	  } 
	}else if(wkz_feature[ii]=="triangle right"){
	  double gradient = (0.5*(pmax[dim-1]-pmin[dim-1]))/(pmax[0]-pmin[0]);
	  double c = pmin[0]*gradient;
	  if(rotated_position[0]>pmin[0]&&rotated_position[dim-1]>pmin[dim-1]-c+rotated_position[0]*gradient&&rotated_position[dim-1]<pmax[dim-1]+c-rotated_position[0]*gradient){
	 /*
	 double tmpvisosity = viscosity;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmpvisosity =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(viscosity/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;

	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);
       highest_weakening = wkz_weakening[ii];*/
       temp+=wkz_heating[ii];
       
	  }
	}else if(wkz_feature[ii]=="triangle left"){
	  double gradient = (0.5*(pmax[dim-1]-pmin[dim-1]))/(pmax[0]-pmin[0]);
	  double c = pmin[0]*gradient;
	  if(rotated_position[0]<pmax[0]&&rotated_position[dim-1]<pmin[dim-1]-c+0.5*(pmax[dim-1]-pmin[dim-1])+rotated_position[0]*gradient&&rotated_position[dim-1]>pmax[dim-1]+c-0.5*(pmax[dim-1]-pmin[dim-1])-rotated_position[0]*gradient){
	 /*
	 double tmpvisosity = viscosity;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmpvisosity =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(viscosity/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;
	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);

       highest_weakening = wkz_weakening[ii];*/
       temp+=wkz_heating[ii];
       
	  }
	}else { /// rectangle
     if(dim==2&&(rotated_position[dim-1]<pmax[dim-1]&&rotated_position[dim-1]>pmin[dim-1])&&(rotated_position[0]<pmax[0]&&rotated_position[0]>pmin[0])){
	 /*
	 double tmpvisosity = viscosity;
	if(wkz_orderx[ii]==-10910||wkz_ordery[ii]==-10910){
	 tmpvisosity =  pow(10,log10(wkz_weakening[ii])*(pow((rotated_position[0]-pmin[0])/(0.5*wkz_length[ii]),2)+pow((rotated_position[dim-1]-pmin[dim-1])/(0.5*wkz_thickness[ii]),2))+log10(viscosity/wkz_weakening[ii]));
	 
	}else{
	  double multiplier=1;
	  
	  double multiplierx=(1-std::pow(std::abs((rotated_position[dim-1]-pmin[dim-1]-0.5*wkz_thickness[ii])/(0.5*wkz_thickness[ii])),wkz_orderx[ii]));
	  double multipliery=(1-std::pow(std::abs((rotated_position[0]-pmin[0]-0.5*wkz_length[ii])/(0.5*wkz_length[ii])),wkz_ordery[ii]));
	  

	  
	  if(multiplierx<multipliery){
	   multiplier = multiplierx;
	  }else{
	  multiplier=multipliery;

	  }
	tmpvisosity = viscosity/(wkz_weakening[ii]*multiplier);
	}
       viscosity = std::max(std::min(tmpvisosity,maximum_eta),minimum_eta);

       highest_weakening = wkz_weakening[ii];*/
       temp+=wkz_heating[ii];
       
     }
     }
       
     }
    }
   ////////////////////////////////////////////// end introduction weakzones //////////////////////////////////////////////

    return temp;
    }


    template <int dim>
    void
    SubductionTemp2<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Subduction temperature");
        {
          prm.declare_entry ("Slab thickness", "80e3",
                             Patterns::Double (0),
                             "The thickness of the slab. "
                             "In meters.");
          prm.declare_entry ("Slab dip", "30",
                             Patterns::Double (0),
                             "The dip of the slab in degrees.");
          prm.declare_entry ("Rotation point", "500e3",
                             Patterns::Double (0),
                             "The point at the surface where the slab rotates from.");
          prm.declare_entry ("Slab length", "300e3",
                             Patterns::Double (0),
                             "The length of the slab in meters measured along the top.");
          prm.declare_entry ("Height to surface", "660e3",
                             Patterns::Double (0),
                             "The distance from the bottom of the model to the surface in meters.");
          prm.declare_entry ("Overriding plate thickness", "100e3",
                             Patterns::Double (0),
                             "The thickness of the overriding plate in meters.");
          prm.declare_entry ("Subducting plate crust thickness", "5e3",
                             Patterns::Double (0),
                             "The thickness of the plate of the subducting plate in meters.");
          prm.declare_entry ("Subduction velocity", "0.1",
                             Patterns::Double (0),
                             "The velocity with which the plate has been subducting in meters per year.");
          prm.declare_entry ("Surface temperature", "273.15",
                             Patterns::Double (0),
                             "The temperature at the surface in degrees Kelvin.");
          prm.declare_entry ("Potential temperature slab on surface", "1553",
                             Patterns::Double (0),
                             "The potiential temperature at the surface used for the adiabat of the subducting slab in degrees Kelvin.");
	  prm.declare_entry ("Potential temperature mantle on surface", "1700",
                             Patterns::Double (0),
                             "The potiential temperature at the surface used for the adiabat of the mantle in degrees Kelvin.");
          prm.declare_entry ("Number of summation expansions", "40",
                             Patterns::Integer (0),
                             "The ammount of summation expansions used in the mckenzie formula for the slab.");
/*          prm.enter_subsection("Function");
          {
            Functions::ParsedFunction<1>::declare_parameters (prm, 1);
          }
          prm.leave_subsection();
*/        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      prm.enter_subsection("Material model");
       {
        prm.enter_subsection("Viscoplastic model");
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
    SubductionTemp2<dim>::parse_parameters (ParameterHandler &prm)
    {
      // we need to get the number of compositional fields here to
      // initialize the function parser. unfortunately, we can't get it
      // via SimulatorAccess from the simulator itself because at the
      // current point the SimulatorAccess hasn't been initialized
      // yet. so get it from the parameter file directly.
      prm.enter_subsection ("Compositional fields");
      const unsigned int n_compositional_fields = prm.get_integer ("Number of fields");
      prm.leave_subsection ();


      prm.enter_subsection ("Geometry model");
      prm.enter_subsection ("Box");
      prm.leave_subsection ();
      prm.leave_subsection ();
     // prm.enter_subsection ("Gravity model");
      //std::string ModelName = prm.get_string("Model name")
      //prm.enter_subsection (ModelName);
      //g = prm.get_double ("Magnitude");
      //prm.leave_subsection ();
      //prm.leave_subsection ();

      prm.enter_subsection ("Material model");
      prm.enter_subsection ("Viscoplastic model");
      {
//Please uncomment the following when the inputfile is correct
      density = prm.get_double ("Reference density");
      Cp = prm.get_double ("Reference specific heat");
      k = prm.get_double ("Thermal conductivity");
      alfa = prm.get_double ("Thermal expansion coefficient");
//////
      /// do stuff for weakzones 
      prm.enter_subsection("Weakzones");
      {
	wkz_n_zones = prm.get_integer ("Number of weakzones");

          if (wkz_n_zones > 0){
          
          //parametes needed for all weakzones
          
          const std::vector<double> n_wkz_heating = Utilities::string_to_double
                                                      (Utilities::split_string_list(prm.get ("List of add to initial temperature")));
          wkz_heating = std::vector<double> (n_wkz_heating.begin(),
                                                         n_wkz_heating.end());
          AssertThrow (wkz_heating.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of add to initial temperature"));
	  
	  
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
      prm.leave_subsection ();
      prm.leave_subsection ();
      
      //prm.enter_subsection("Boundary temperature model");
      //prm.enter_subsection("MckenzieBoundary");
      //std::cout << Ttop << std::endl;
      //prm.leave_subsection();
      //prm.leave_subsection();
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Subduction temperature");
        {
         l = prm.get_double ("Slab thickness");
         //coordrot = prm.get_double ("Coordinate rotation");
         dip = prm.get_double ("Slab dip");
         Px = prm.get_double ("Rotation point");
         d = prm.get_double ("Slab length");
         depth = prm.get_double ("Height to surface");
         lc3 = prm.get_double ("Overriding plate thickness");
         lcr2 = prm.get_double ("Subducting plate crust thickness");
         v = prm.get_double ("Subduction velocity");
	Ttop = prm.get_double("Surface temperature");
	 T_pot = prm.get_double ("Potential temperature slab on surface");
         pTsm = prm.get_double ("Potential temperature mantle on surface");
         n_sum = prm.get_integer ("Number of summation expansions");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
/*      if (n_compositional_fields > 0)
         {
          prm.enter_subsection("Compositional initial conditions");
          {   
           prm.enter_subsection("Function");
           {
            function.reset (new Functions::ParsedFunction<2>(n_compositional_fields));
            function->parse_parameters (prm);
           }
           prm.leave_subsection();
          }
          prm.leave_subsection();
         }*/
   }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(SubductionTemp2,
                                       "subduction temperature2",
                                       "Temperature is prescribed as a linear profile in the lithosphere, "
                                       "adiabat in the mantle and according to McKenzie 1970 in the slab. "
                                       "Slab properties (e.g. dip) can be specified in the input file.")
  }
}

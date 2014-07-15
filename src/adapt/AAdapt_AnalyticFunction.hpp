//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_ANALYTICFUNCTION_HPP
#define AADAPT_ANALYTICFUNCTION_HPP

#include <string>

// Random and Gaussian number distribution
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include "Teuchos_Array.hpp"

namespace AAdapt {

// generate seed convenience function
long seedgen(int worksetID);

// Base class for initial condition functions
class AnalyticFunction {
  public:
    virtual ~AnalyticFunction() {}
    virtual void compute(double* x, const double* X) = 0;
};

// Factory method to build functions based on a string name
Teuchos::RCP<AnalyticFunction> createAnalyticFunction(
  std::string name, int neq, int numDim,
  Teuchos::Array<double> data);

// Below is a library of intial condition functions

class ConstantFunction : public AnalyticFunction {
  public:
    ConstantFunction(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class ConstantFunctionPerturbed : public AnalyticFunction {
  public:
    ConstantFunctionPerturbed(int neq_, int numDim_, int worksetID,
                              Teuchos::Array<double> const_data_, Teuchos::Array<double> pert_mag_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
    Teuchos::Array<double> pert_mag;

    // random number generator convenience function
    double udrand(double lo, double hi);

};

class ConstantFunctionGaussianPerturbed : public AnalyticFunction {
  public:
    ConstantFunctionGaussianPerturbed(int neq_, int numDim_, int worksetID,
                                      Teuchos::Array<double> const_data_, Teuchos::Array<double> pert_mag_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
    Teuchos::Array<double> pert_mag;

    boost::mt19937 rng;
    Teuchos::Array<Teuchos::RCP<boost::normal_distribution<double> > > nd;
    Teuchos::Array < Teuchos::RCP < boost::variate_generator < boost::mt19937&,
            boost::normal_distribution<double> > > > var_nor;

};

class GaussSin : public AnalyticFunction {
  public:
    GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class GaussCos : public AnalyticFunction {
  public:
    GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class LinearY : public AnalyticFunction {
  public:
    LinearY(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class GaussianPress : public AnalyticFunction {
  public:
    GaussianPress(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class SinCos : public AnalyticFunction {
  public:
    SinCos(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class TaylorGreenVortex : public AnalyticFunction {
  public:
    TaylorGreenVortex(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AcousticWave : public AnalyticFunction {
  public:
    AcousticWave(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasScharDensity : public AnalyticFunction {
  public:
    AerasScharDensity(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXScalarAdvection : public AnalyticFunction {
  public:
    AerasXScalarAdvection(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostatic : public AnalyticFunction {
  public:
    AerasXZHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasHydrostatic : public AnalyticFunction {
  public:
    AerasHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasHeaviside : public AnalyticFunction {
  public:
    AerasHeaviside(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasCosineBell : public AnalyticFunction {
  public:
    AerasCosineBell(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasZonalFlow : public AnalyticFunction {
  public:
     AerasZonalFlow(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x

    Teuchos::Array<double> data;
};
class AerasTC5Init : public AnalyticFunction {
  public:
     AerasTC5Init(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);

  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    
    Teuchos::Array<double> data;
};
    
class AerasTC3Init : public AnalyticFunction {
    public:
        AerasTC3Init(int neq_, int spatialDim_, Teuchos::Array<double> data_);
        void compute(double* x, const double* X);
    
    private:
        int spatialDim; // size of coordinate vector X
        int neq;    // size of solution vector x
    
        Teuchos::Array<double> data;
    
    private:
        double bx(const double x); //an indicator function
    
        double ucomponent(const double lon); //unrotated u-comp in TC3
    
        void rotate(const double lon, const double lat, const double alpha, double& rotlon, double& rotlat); //obtains rotated lon lat for TC3.
    
        double earthRadius; //Earth radius
        double testDuration; // =12 days, in seconds
        double myPi; // a local copy of pi
        double u0; //a u-comp velocity multiplier, based on Williamson1992
};
    
class AerasTCGalewskyInit : public AnalyticFunction {
    public:
        AerasTCGalewskyInit(int neq_, int spatialDim_, Teuchos::Array<double> data_);
        void compute(double* x, const double* X);
        
    private:
        int spatialDim; // size of coordinate vector X
        int neq;    // size of solution vector x
        
        Teuchos::Array<double> data;
        
    private:
    
        double ucomponent(const double lon); //
        double hperturb(const double lon, const double lat); //
    
        double earthRadius; //Earth radius
        //double testDuration; // =12 days, in seconds, should be a param from data_
        double myPi; // a local copy of pi
    
        double phi0;
        double phi1;
        const double umax = 80;
        double en;

        const double h0 = 10000.0;
        double phi2;
        const double al = 1./3.;
        const double beta =  1./15.;
        const double hhat = 120.; //meters
};
    
class AerasPlanarCosineBell : public AnalyticFunction {
  public:
    AerasPlanarCosineBell(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasRossbyHaurwitzWave : public AnalyticFunction {
  public:
    AerasRossbyHaurwitzWave(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x

    Teuchos::Array<double> data;
};

}

#endif

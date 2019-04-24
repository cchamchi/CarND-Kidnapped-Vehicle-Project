/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#define EPS 0.00001

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  if(is_initialized) {
    return;
  } 
  // TODO: Set the number of particles
  num_particles = 100;
  // I will use weights at resampling
  weights.resize(num_particles);
  //initialize stds
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  //create normal distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  //Generate particles with normal distribution with mean on GPS positions.
  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  //for fast calcuation, prepair Normal distributions with zero mean at outer loop
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i < num_particles; i++) {
    if(fabs(yaw_rate) < EPS) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    //just add each position to zero mean noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); i++) { 
    // Initialize min distance as a really big number.
    double min_distance = std::numeric_limits<double>::max();
    // Initialize the found map in something not possible.
    int mapId = -1;
    // find nearest id in predicted
    for (unsigned j = 0; j < predicted.size(); j++ ) { 
      double distance=dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
      if ( distance < min_distance ) {
        min_distance = distance;
        mapId = predicted[j].id;
      }
    }
    // Update the observation identifier.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  for (int i = 0; i < num_particles; i++) {
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;


    Particle particle = particles[i];
    // Find landmarks in particle's sensor range.
    // Used Squared value not to use square root
    double squared_range = sensor_range * sensor_range;
    vector<LandmarkObs> landmarks_in_range;
 
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs landmark;
      landmark.x = map_landmarks.landmark_list[j].x_f;
      landmark.y = map_landmarks.landmark_list[j].y_f;
      landmark.id = map_landmarks.landmark_list[j].id_i;
      double dX = particle.x - landmark.x;
      double dY = particle.y - landmark.y;
      if ( dX*dX + dY*dY <= squared_range ) {
        landmarks_in_range.push_back(landmark);
      }
    }

    // transform each observation marker from the vehicle's coordinates 
    // to the map's coordinates, with respect to our particle.
    vector<LandmarkObs> transformed_observations;
    for(unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs tranformed;
      tranformed.x = cos(particle.theta)*observations[j].x - sin(particle.theta)*observations[j].y + particle.x;
      tranformed.y = sin(particle.theta)*observations[j].x + cos(particle.theta)*observations[j].y + particle.y;
      tranformed.id = observations[j].id;
      transformed_observations.push_back(tranformed);
    }

    // Observation association to landmark.
    dataAssociation(landmarks_in_range, transformed_observations);

    // Reseting weight.
    particles[i].weight = 1.0;
    // Calculate weights.
    for(unsigned int j = 0; j < transformed_observations.size(); j++) {
      double observationX = transformed_observations[j].x;
      double observationY = transformed_observations[j].y;

      int minIndex=0;
      double landmarkX, landmarkY;

      double min_distance = std::numeric_limits<double>::max();
      for (unsigned int j=0; j<landmarks_in_range.size(); ++j){
        LandmarkObs pred = landmarks_in_range[j];
        double distance = dist(pred.x, pred.y, observationX, observationY);

        if (distance<min_distance){
          min_distance = distance;
          landmarkX = pred.x;
          landmarkY = pred.y;
          minIndex = j;
        }
      }      
      

      // Calculating weight.
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;

      double weight = ( 1/(2*M_PI*sigma_x*sigma_y)) * exp( -( dX*dX/(2*sigma_x*sigma_x) + (dY*dY/(2*sigma_y*sigma_y)) ) );
      if (weight == 0) {
        particles[i].weight *= EPS;
      } else {
        particles[i].weight *= weight;
      }
      weights[i] = particles[i].weight;
      
      associations.push_back(minIndex+1);
      sense_x.push_back(observationX);
      sense_y.push_back(observationY);

    }
    // Optional message data used for debugging particle's sensing 
    //   and associations
    //SetAssociations(particles[i], associations, sense_x, sense_y);
  }


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    std::vector<Particle> new_particles;
    std::default_random_engine gen;

    std::uniform_int_distribution<> int_dist(0, num_particles - 1);
    int index = int_dist(gen);

    double max_weight = *std::max_element(weights.begin(), weights.end());
    std::uniform_real_distribution<double> real_dist(0.0,2*max_weight);

    double beta = 0;
    for (int i = 0; i < num_particles; i++) {
        beta += real_dist(gen);

        while(beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
  
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
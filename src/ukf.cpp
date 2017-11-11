#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_.fill(0.0);
    P_(0, 0) = 1.0;
    P_(1, 1) = 1.0;
    P_(2, 2) = 1.0;
    P_(3, 3) = 1.0;
    P_(4, 4) = 1.0;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1.0;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */
    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    //process noise covariance matrix
    Q_ = MatrixXd::Zero(2, 2);
    Q_(0, 0) = std_a_ * std_a_;
    Q_(1, 1) = std_yawdd_ * std_yawdd_;

    //laser measurement noise covariance matrix
    R_laser_ = MatrixXd::Zero(2, 2);
    R_laser_(0, 0) = std_laspx_*std_laspx_;
    R_laser_(1, 1) = std_laspy_*std_laspy_;

    //radar measurement noise covariance matrix
    R_radar_ = MatrixXd::Zero(3, 3);
    R_radar_(0, 0) = std_radr_*std_radr_;
    R_radar_(1, 1) = std_radphi_*std_radphi_;
    R_radar_(2, 2) = std_radrd_*std_radrd_;

    //augmented state covariance matrix
    P_aug_ = MatrixXd(n_aug_, n_aug_);
    P_aug_.fill(0.0);

    //augmented state vector
    x_aug_ = VectorXd(n_aug_);

    //set weights
    weights_ = VectorXd(2*n_aug_+1);
    weights_(0) = lambda_/(lambda_+n_aug_);
    for (int i = 1; i < 2*n_aug_+1; i++){
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }

    is_initialized_ = false;
}

UKF::~UKF() {}

/**@brief process measurement
 *
 * process measurement: predict state; update the state;
 * @param {MeasurementPackage} meas_package The latest measurement data of either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  //if don't use laser
  if(MeasurementPackage::LASER == meas_package.sensor_type_ && !use_laser_){
    return;
  }

  //if don't use raser
  if(MeasurementPackage::RADAR == meas_package.sensor_type_ && !use_radar_){
    return;
  }

  if(!is_initialized_){
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    if(MeasurementPackage::RADAR == meas_package.sensor_type_){
      const double rho = meas_package.raw_measurements_[0];
      const double phi = meas_package.raw_measurements_[1];
      const double rho_dot = meas_package.raw_measurements_[2];
      x_ << rho*cos(phi), rho*sin(phi), rho_dot, 0.0, 0.0;
    }

    if(MeasurementPackage::LASER == meas_package.sensor_type_){
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
    }
    return;
  }

  const double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;

  //predict state
  Prediction(delta_t);

  //update state
  if(MeasurementPackage::RADAR == meas_package.sensor_type_){
    UpdateRadar(meas_package);
  }
  if(MeasurementPackage::LASER == meas_package.sensor_type_){
    UpdateLidar(meas_package);
  }

  //update timestamp
  time_us_ = meas_package.timestamp_;
}

/**@brief predict state and covariance;
 *
 * Predicts sigma points, the state, and the state covariance matrix:
 * 1. Generate augmented sigma points;
 * 2. Predict sigma points;
 * 3. Predict mean and covariance;
 * @param {double} delta_t the change in time (in seconds) between the last measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //generate augmented sigma points;
  MatrixXd X_aug_sig;
  GenerateAugmentedSigmaPoints(&X_aug_sig);

  //predict sigma points
  PredictSigmaPoints(X_aug_sig, delta_t, &Xsig_pred_);

  //predict mean and covariance;
  PredictMeanAndCovariance();
}


/**@brief update the state and covariance by a laser measurement
 *
 * predict measurement; calculate state and covariance; calculate NIS;
 * @param meas_package [IN]: the lidar measurement;
 * @note the update of lidar could use standard kalman filter or UKF;
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
#ifdef DEBUG_LIDAR_KF
    //measurement matrix;
    MatrixXd H = MatrixXd(2, 5);
    H.fill(0.0);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    //predict measurement;
    VectorXd z_pred = H*x_;
    VectorXd y = meas_package.raw_measurements_ - z_pred;
    MatrixXd S = H*P_*H.transpose() + R_laser_;
    MatrixXd K = P_*H.transpose()*S.inverse();
    MatrixXd I = MatrixXd::Identity(5, 5);
    //update state;
    x_ = x_ + K*y;
    P_ = (I-K*H)*P_;
#endif

#ifdef DEBUG_LIDAR_UKF
    MatrixXd Zsig;//predicted sigma points
    VectorXd z_pred; // mean of predicted sigma points;
    MatrixXd S; //covariance of predicted sigma points

    PredictLidarMeasurement(&Zsig, &z_pred, &S);

    UpdateLidarState(Zsig, z_pred, S, meas_package);

#endif

    NIS_laser_ = (meas_package.raw_measurements_-z_pred).transpose()*S.inverse()*(meas_package.raw_measurements_-z_pred);
#ifdef DEBUG_OUTPUT
    cout << "NIS_laser:" << "\t" << NIS_laser_ << endl;
#endif
}

/**@brief update radar state and covariance
 *
 * predict radar measurement and covariance based on sigma points; update state and covariance by UKF, calculate NIS;
 * @param meas_package [IN]: radar measurement
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
    MatrixXd Zsig;//predicted sigma points
    VectorXd z_pred; // mean of predicted sigma points;
    MatrixXd S; //covariance of predicted sigma points

    PredictRadarMeasurement(&Zsig, &z_pred, &S);

    UpdateRadarState(Zsig, z_pred, S, meas_package);

    NIS_radar_ = (meas_package.raw_measurements_-z_pred).transpose()*S.inverse()*(meas_package.raw_measurements_-z_pred);
#ifdef DEBUG_OUTPUT
    cout << "NIS_radar:" << "\t" << NIS_radar_ << endl;
#endif
}


/**brief generate augmented sigma points;
 *
 * @param Xsig_out [OUT]: the augmented sigma points;
 */
void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_out){
    MatrixXd X_sig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

    //create augmented mean state
    x_aug_.head(n_x_) = x_;
    x_aug_(5) = 0.0;
    x_aug_(6) = 0.0;

    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_.bottomRightCorner(2, 2) = Q_;

    //create square root matrix
    MatrixXd L = P_aug_.llt().matrixL();

    //create augmented sigma points
    X_sig_aug.col(0)  = x_aug_;
    for (int i = 0; i< n_aug_; i++)
    {
    X_sig_aug.col(i+1)= x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    X_sig_aug.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
    }

    *Xsig_out = X_sig_aug;
}



/**@brief predict sigma points
 *
 * predict sigma points
 * @param X_aug_sig_out [IN]: the augmented sigma points;
 * @param delta_t [IN]: the delta time between the current and the previous timestamp;
 * @param Xsig_pred_out [OUT]: the predicted sigma points;
 */
void UKF::PredictSigmaPoints(const MatrixXd& X_aug_sig, const double delta_t, MatrixXd* X_pred_sig_out){
  MatrixXd X_pred_sig = MatrixXd(n_x_, 2*n_aug_+1);

  for (int i = 0; i < 2*n_aug_+1; i++){
    double p_x = X_aug_sig(0,i);
    double p_y = X_aug_sig(1,i);
    double v = X_aug_sig(2,i);
    double yaw = X_aug_sig(3,i);
    double yawd = X_aug_sig(4,i);
    double nu_a = X_aug_sig(5,i);
    double nu_yawdd = X_aug_sig(6,i);

    double pred_p_x, pred_p_y, pred_v, pred_yaw, pred_yawd;

    if (fabs(yawd) > 1e-3){
      pred_p_x = p_x + v * (sin(yaw + yawd * delta_t) - sin(yaw)) / yawd;
      pred_p_y = p_y + v * (-cos(yaw + yawd * delta_t) + cos(yaw)) / yawd;
    }
    else{
      pred_p_x = p_x + v * cos(yaw) * delta_t;
      pred_p_y = p_y + v * sin(yaw) * delta_t;
    }
    pred_v = v;
    pred_yaw = yaw + yawd * delta_t;
    pred_yawd = yawd;

    //add noise
    pred_p_x += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    pred_p_y += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    pred_v += delta_t * nu_a;
    pred_yaw += 0.5 * delta_t * delta_t * nu_yawdd;
    pred_yawd += delta_t * nu_yawdd;

    X_pred_sig(0, i) = pred_p_x;
    X_pred_sig(1, i) = pred_p_y;
    X_pred_sig(2, i) = pred_v;
    X_pred_sig(3, i) = pred_yaw;
    X_pred_sig(4, i) = pred_yawd;
  }
  *X_pred_sig_out = X_pred_sig;
}


/**@brief calculate mean and covariance of predicted sigma points
 *
 */
void UKF::PredictMeanAndCovariance() {
  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++){
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {  //iterate over sigma points

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}


/**@brief predict radar measurement
 *
 * predict radar measurement based on the sigma points, calculate the mean and covariance of predicted sigma points of z
 * @param Zsig_out [OUT]: predicted sigma points of z;
 * @param z_pred_out [OUT]: mean of predicted sigma points of z;
 * @param S_out [OUT]: covariance of predicted sigma points of z;
 */
void UKF::PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_pred_out, MatrixXd* S_out){
  //transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd(3, 2*n_aug_+1);
  for (int i = 0; i < 2*n_aug_+1; i++){
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //calculate mean of predicted measurement
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate covariance matrix
  MatrixXd S = MatrixXd(3, 3);
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_radar_;

  *Zsig_out = Zsig;
  *z_pred_out = z_pred;
  *S_out = S;
}


/**@brief update radar state and covariance
 *
 * update state and covariance:
 * @param Zsig [IN]: The predicted sigma points of z;
 * @param z_pred [IN]: mean of predicted sigma points of z;
 * @param S [IN]: covariance of predicted sigma points of z;
 * @param meas_package [IN]: radar measurement;
 */
void UKF::UpdateRadarState(const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S, const MeasurementPackage& meas_package){
    //calculate Tc: Cross-correlation between sigma points in state space and measurement space;
    MatrixXd Tc = MatrixXd(n_x_, 3);
    Tc.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual z;
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K*z_diff;
    P_ = P_ - K*S*K.transpose();
}


/**@brief predict lidar measurement
 *
 * predict lidar measurement based on the sigma points, calculate the mean and covariance of predicted sigma points of z
 * @param Zsig_out [OUT]: predicted sigma points of z;
 * @param z_pred_out [OUT]: mean of predicted sigma points of z;
 * @param S_out [OUT]: covariance of predicted sigma points of z;
 */
void UKF::PredictLidarMeasurement(MatrixXd* Zsig_out, VectorXd* z_pred_out, MatrixXd* S_out){
    int n_z = 2; //set measurement dimension
    //transform sigma points into measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
    for (int i = 0; i < 2*n_aug_+1; i++){
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0,i) = p_x;
        Zsig(1,i) = p_y;
    }

    //calculate mean of predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //calculate covariance matrix
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_laser_;

    *Zsig_out = Zsig;
    *z_pred_out = z_pred;
    *S_out = S;
}



/**@brief update lidar state and covariance
 *
 * update state and covariance:
 * @param Zsig [IN]: The predicted sigma points of z;
 * @param z_pred [IN]: mean of predicted sigma points of z;
 * @param S [IN]: covariance of predicted sigma points of z;
 * @param meas_package [IN]: radar measurement;
 */
void UKF::UpdateLidarState(const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S, const MeasurementPackage& meas_package){
    //calculate Tc: Cross-correlation between sigma points in state space and measurement space;
    MatrixXd Tc = MatrixXd(n_x_, 2);
    Tc.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //kalman gain K;
    MatrixXd K = Tc * S.inverse();
    //residual z;
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K*z_diff;
    P_ = P_ - K*S*K.transpose();
}

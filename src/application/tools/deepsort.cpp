#include "deepsort.hpp"

#include <vector>
#include <set>
#include <algorithm>
#include <utility>
#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/LU"

namespace DeepSORT {

    struct BBoxXYAH{
        int center_x, center_y;   
        float aspect_ratio;       
        int height;               

        BBoxXYAH() = default;
        BBoxXYAH(const Box &box) {
            const auto center = box.center();
            center_x = center.x;
            center_y = center.y;
            height = box.height();
            aspect_ratio = box.width() / height;
        }
    };

    static float chi2inv95_2[] = {
        3.8415f,
        5.9915f,
        7.8147f,
        9.4877f,
        11.070f,
        12.592f,
        14.067f,
        15.507f,
        16.919
    };

    static float distance(const Box &box, const Box &box2) {
        const auto center = box.center();
        const auto center2 = box2.center();
        return hypot(center.x - center2.x, center.y - center2.y);
    }

    class HungarianAlgorithm
    {
    public:
        enum TMethod
        {
            optimal,
            many_forbidden_assignments,
            without_forbidden_assignments
        };

    public:
        HungarianAlgorithm(){}
        ~HungarianAlgorithm(){}

        double Solve(std::vector<std::vector<double> >& DistMatrix, std::vector<int>& Assignment)
        {
            unsigned int nRows = DistMatrix.size();
            unsigned int nCols = DistMatrix[0].size();

            std::vector<double> distMatrixIn(nRows * nCols);
            std::vector<int> assignment(nRows);
            double cost = 0.0;

            for (unsigned int i = 0; i < nRows; i++)
                for (unsigned int j = 0; j < nCols; j++)
                    distMatrixIn[i + nRows * j] = DistMatrix[i][j];
            
            // call solving function
            assignmentoptimal(assignment.data(), &cost, distMatrixIn.data(), nRows, nCols);

            Assignment.clear();
            for (unsigned int r = 0; r < nRows; r++)
                Assignment.push_back(assignment[r]);

            return cost;
        }

    private:
        void assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
        {
            double *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
            bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
            int nOfElements, minDim, row, col;

            /* initialization */
            *cost = 0;
            for (row = 0; row<nOfRows; row++)
                assignment[row] = -1;

            /* generate working copy of distance Matrix */
            /* check if all matrix elements are positive */
            nOfElements = nOfRows * nOfColumns;
            distMatrix = (double *)malloc(nOfElements * sizeof(double));
            distMatrixEnd = distMatrix + nOfElements;

            for (row = 0; row<nOfElements; row++)
            {
                value = distMatrixIn[row];
                if (value < 0)
                    std::cerr << "All matrix elements have to be non-negative." << std::endl;
                distMatrix[row] = value;
            }


            /* memory allocation */
            coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
            coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
            starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
            primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
            newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

            /* preliminary steps */
            if (nOfRows <= nOfColumns)
            {
                minDim = nOfRows;

                for (row = 0; row<nOfRows; row++)
                {
                    /* find the smallest element in the row */
                    distMatrixTemp = distMatrix + row;
                    minValue = *distMatrixTemp;
                    distMatrixTemp += nOfRows;
                    while (distMatrixTemp < distMatrixEnd)
                    {
                        value = *distMatrixTemp;
                        if (value < minValue)
                            minValue = value;
                        distMatrixTemp += nOfRows;
                    }

                    /* subtract the smallest element from each element of the row */
                    distMatrixTemp = distMatrix + row;
                    while (distMatrixTemp < distMatrixEnd)
                    {
                        *distMatrixTemp -= minValue;
                        distMatrixTemp += nOfRows;
                    }
                }

                /* Steps 1 and 2a */
                for (row = 0; row<nOfRows; row++)
                    for (col = 0; col<nOfColumns; col++)
                        if (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON)
                            if (!coveredColumns[col])
                            {
                                starMatrix[row + nOfRows*col] = true;
                                coveredColumns[col] = true;
                                break;
                            }
            }
            else /* if(nOfRows > nOfColumns) */
            {
                minDim = nOfColumns;

                for (col = 0; col<nOfColumns; col++)
                {
                    /* find the smallest element in the column */
                    distMatrixTemp = distMatrix + nOfRows*col;
                    columnEnd = distMatrixTemp + nOfRows;

                    minValue = *distMatrixTemp++;
                    while (distMatrixTemp < columnEnd)
                    {
                        value = *distMatrixTemp++;
                        if (value < minValue)
                            minValue = value;
                    }

                    /* subtract the smallest element from each element of the column */
                    distMatrixTemp = distMatrix + nOfRows*col;
                    while (distMatrixTemp < columnEnd)
                        *distMatrixTemp++ -= minValue;
                }

                /* Steps 1 and 2a */
                for (col = 0; col<nOfColumns; col++)
                    for (row = 0; row<nOfRows; row++)
                        if (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON)
                            if (!coveredRows[row])
                            {
                                starMatrix[row + nOfRows*col] = true;
                                coveredColumns[col] = true;
                                coveredRows[row] = true;
                                break;
                            }
                for (row = 0; row<nOfRows; row++)
                    coveredRows[row] = false;

            }

            /* move to step 2b */
            step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

            /* compute cost and remove invalid assignments */
            computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

            /* free allocated memory */
            free(distMatrix);
            free(coveredColumns);
            free(coveredRows);
            free(starMatrix);
            free(primeMatrix);
            free(newStarMatrix);

            return;
        }

        void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
        {
            int row, col;

            for (row = 0; row<nOfRows; row++)
                for (col = 0; col<nOfColumns; col++)
                    if (starMatrix[row + nOfRows*col])
                    {
        #ifdef ONE_INDEXING
                        assignment[row] = col + 1; /* MATLAB-Indexing */
        #else
                        assignment[row] = col;
        #endif
                        break;
                    }
        }

        void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows)
        {
            int row, col;

            for (row = 0; row<nOfRows; row++)
            {
                col = assignment[row];
                if (col >= 0)
                    *cost += distMatrix[row + nOfRows*col];
            }
        }

        void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
        {
            bool *starMatrixTemp, *columnEnd;
            int col;

            /* cover every column containing a starred zero */
            for (col = 0; col<nOfColumns; col++)
            {
                starMatrixTemp = starMatrix + nOfRows*col;
                columnEnd = starMatrixTemp + nOfRows;
                while (starMatrixTemp < columnEnd){
                    if (*starMatrixTemp++)
                    {
                        coveredColumns[col] = true;
                        break;
                    }
                }
            }

            /* move to step 3 */
            step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }

        void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
        {
            int col, nOfCoveredColumns;

            /* count covered columns */
            nOfCoveredColumns = 0;
            for (col = 0; col<nOfColumns; col++)
                if (coveredColumns[col])
                    nOfCoveredColumns++;

            if (nOfCoveredColumns == minDim)
            {
                /* algorithm finished */
                buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
            }
            else
            {
                /* move to step 3 */
                step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
            }

        }

        void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
        {
            bool zerosFound;
            int row, col, starCol;

            zerosFound = true;
            while (zerosFound)
            {
                zerosFound = false;
                for (col = 0; col<nOfColumns; col++)
                    if (!coveredColumns[col])
                        for (row = 0; row<nOfRows; row++)
                            if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON))
                            {
                                /* prime zero */
                                primeMatrix[row + nOfRows*col] = true;

                                /* find starred zero in current row */
                                for (starCol = 0; starCol<nOfColumns; starCol++)
                                    if (starMatrix[row + nOfRows*starCol])
                                        break;

                                if (starCol == nOfColumns) /* no starred zero found */
                                {
                                    /* move to step 4 */
                                    step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                                    return;
                                }
                                else
                                {
                                    coveredRows[row] = true;
                                    coveredColumns[starCol] = false;
                                    zerosFound = true;
                                    break;
                                }
                            }
            }

            /* move to step 5 */
            step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }

        void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
        {
            int n, starRow, starCol, primeRow, primeCol;
            int nOfElements = nOfRows*nOfColumns;

            /* generate temporary copy of starMatrix */
            for (n = 0; n<nOfElements; n++)
                newStarMatrix[n] = starMatrix[n];

            /* star current zero */
            newStarMatrix[row + nOfRows*col] = true;

            /* find starred zero in current column */
            starCol = col;
            for (starRow = 0; starRow<nOfRows; starRow++)
                if (starMatrix[starRow + nOfRows*starCol])
                    break;

            while (starRow<nOfRows)
            {
                /* unstar the starred zero */
                newStarMatrix[starRow + nOfRows*starCol] = false;

                /* find primed zero in current row */
                primeRow = starRow;
                for (primeCol = 0; primeCol<nOfColumns; primeCol++)
                    if (primeMatrix[primeRow + nOfRows*primeCol])
                        break;

                /* star the primed zero */
                newStarMatrix[primeRow + nOfRows*primeCol] = true;

                /* find starred zero in current column */
                starCol = primeCol;
                for (starRow = 0; starRow<nOfRows; starRow++)
                    if (starMatrix[starRow + nOfRows*starCol])
                        break;
            }

            /* use temporary copy as new starMatrix */
            /* delete all primes, uncover all rows */
            for (n = 0; n<nOfElements; n++)
            {
                primeMatrix[n] = false;
                starMatrix[n] = newStarMatrix[n];
            }
            for (n = 0; n<nOfRows; n++)
                coveredRows[n] = false;

            /* move to step 2a */
            step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }

        void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
        {
            double h, value;
            int row, col;

            /* find smallest uncovered element h */
            h = DBL_MAX;
            for (row = 0; row<nOfRows; row++)
                if (!coveredRows[row])
                    for (col = 0; col<nOfColumns; col++)
                        if (!coveredColumns[col])
                        {
                            value = distMatrix[row + nOfRows*col];
                            if (value < h)
                                h = value;
                        }

            /* add h to each covered row */
            for (row = 0; row<nOfRows; row++)
                if (coveredRows[row])
                    for (col = 0; col<nOfColumns; col++)
                        distMatrix[row + nOfRows*col] += h;

            /* subtract h from each uncovered column */
            for (col = 0; col<nOfColumns; col++)
                if (!coveredColumns[col])
                    for (row = 0; row<nOfRows; row++)
                        distMatrix[row + nOfRows*col] -= h;

            /* move to step 3 */
            step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }
    };


    TrackerConfig::TrackerConfig(){
        
        float std_weight_position_ = 1 / 20.f;
        float std_weight_velocity_ = 1 / 160.f;
        float initiate_state[] = {
            2.0f * std_weight_position_,
            2.0f * std_weight_position_,
            1e-2,
            2.0f * std_weight_position_,
            10.0f * std_weight_velocity_,
            10.0f * std_weight_velocity_,
            1e-5,
            10.0f * std_weight_velocity_,
        };

        float noise[] = {
            std_weight_position_,
            std_weight_position_,
            1e-1,
            std_weight_position_
        };

        float per_frame_motion[] = {
            std_weight_position_,
            std_weight_position_,
            1e-2,
            std_weight_position_,
            std_weight_velocity_,
            std_weight_velocity_,
            1e-5,
            std_weight_velocity_,
        };
        memcpy(this->initiate_state, initiate_state, sizeof(initiate_state));
        memcpy(this->noise, noise, sizeof(noise));
        memcpy(this->per_frame_motion, per_frame_motion, sizeof(per_frame_motion));
    }

    void TrackerConfig::set_initiate_state(const std::vector<float>& values){
        if(values.size() != 8){
            printf("set_initiate_state failed, Values.size(%d0) != 8\n", values.size());
            return;
        }
        memcpy(this->initiate_state, values.data(), sizeof(this->initiate_state));
    }

    void TrackerConfig::set_per_frame_motion(const std::vector<float>& values){
        if(values.size() != 8){
            printf("set_per_frame_motion failed, Values.size(%d0) != 8\n", values.size());
            return;
        }
        memcpy(this->per_frame_motion, values.data(), sizeof(this->per_frame_motion));
    }

    void TrackerConfig::set_noise(const std::vector<float>& values){
        if(values.size() != 4){
            printf("set_noise failed, Values.size(%d0) != 4\n", values.size());
            return;
        }
        memcpy(this->noise, values.data(), sizeof(this->noise));
    }

    class KalmanFilter
    {
    public:
        KalmanFilter(const TrackerConfig& config):config_(config) {
            /* 匀速直线运动 */
            motion_mat_ = Eigen::Matrix<float, 8, 8>::Identity(8, 8);
            for (int i = 0; i < 4; ++i) {
                motion_mat_(i, 4 + i) = 1;
            }
            update_mat_ = Eigen::Matrix<float, 4, 8>::Identity(4, 8);
        }
        ~KalmanFilter() {

        }

        void project(const Eigen::Matrix<float, 8, 1> &mean, 
                    const Eigen::Matrix<float, 8, 8> &covariance,
                    Eigen::Matrix<float, 4, 1> &mean_ret,
                    Eigen::Matrix<float, 4, 4> &covariance_ret) {
            Eigen::Matrix<float, 4, 1> std_vel;
            
            /* 测量噪声标准差 */
            // std_vel << std_weight_position_ * mean(3, 0),
            //         std_weight_position_ * mean(3, 0),
            //         1e-1,
            //         std_weight_position_ * mean(3, 0);
            std_vel <<  config_.noise[0] * mean(3, 0),
                        config_.noise[1] * mean(3, 0),
                        config_.noise[2],
                        config_.noise[3] * mean(3, 0);
            std_vel = std_vel.array().pow(2).matrix();
            Eigen::Matrix<float, 4, 4> innovation_cov(std_vel.asDiagonal());

            mean_ret = update_mat_ * mean;
            covariance_ret = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
        }

        /**
         * @brief 马氏距离计算
         * 
         * @param mean 
         * @param covariance 
         * @param boxah 
         * @param only_position 
         * @return float 
         */
        float ma_distance(const Eigen::Matrix<float, 8, 1> &mean, 
                        const Eigen::Matrix<float, 8, 8> &covariance,
                        const BBoxXYAH &boxah,
                        bool only_position =  false) {
            // 
            Eigen::Matrix<float, 4, 1> mean_ret;
            Eigen::Matrix<float, 4, 4> covariance_ret;
            this->project(mean, covariance, mean_ret, covariance_ret);

            float squared_maha = 0;
            if (only_position) {

            }
            else {
                auto cholesky_factor = covariance_ret.llt();
                Eigen::Matrix<float, 4, 1> matrix_boxah;
                matrix_boxah << boxah.center_x, boxah.center_y, boxah.aspect_ratio, boxah.height;
                auto d = matrix_boxah - mean_ret;
                auto z = cholesky_factor.solve(d);
                squared_maha = z.array().pow(2).sum();

                Eigen::LLT<Eigen::Matrix<float, 4, 4>> a_factor(covariance_ret);
                Eigen::Matrix<float, 4, 4> a = a_factor.matrixL();
                Eigen::Map<Eigen::MatrixXf> a_map(a.data(), a.rows(), a.cols());
                auto a_inv = a_map.inverse();
                auto zz = d.transpose() * a_inv;
                squared_maha = zz.array().pow(2).sum();
            }

            return squared_maha;
        }

        void predict(Eigen::Matrix<float, 8, 1> &mean, 
                    Eigen::Matrix<float, 8, 8> &covariance) {
            Eigen::Matrix<float, 8, 1> std_pos_vel;

            /* 预测下一步所在位置，那么std_pos则是模型对下一步预测的标准差。可以认为是一帧运动了多少 */
            // std_pos_vel << std_weight_position_ * mean(3, 0),
            //             std_weight_position_ * mean(3, 0),
            //             1e-2,
            //             std_weight_position_ * mean(3, 0),

            //             std_weight_velocity_ * mean(3, 0),
            //             std_weight_velocity_ * mean(3, 0),
            //             1e-5,
            //             std_weight_velocity_ * mean(3, 0);
            std_pos_vel <<  config_.per_frame_motion[0] * mean(3, 0),
                            config_.per_frame_motion[1] * mean(3, 0),
                            config_.per_frame_motion[2],
                            config_.per_frame_motion[3] * mean(3, 0),
                            config_.per_frame_motion[4] * mean(3, 0),
                            config_.per_frame_motion[5] * mean(3, 0),
                            config_.per_frame_motion[6],
                            config_.per_frame_motion[7] * mean(3, 0);
            std_pos_vel = std_pos_vel.array().pow(2).matrix();
            Eigen::Matrix<float, 8, 8> motion_cov(std_pos_vel.asDiagonal());

            mean = motion_mat_ * mean;
            covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
        }

        void update(const BBoxXYAH &boxah,
                    Eigen::Matrix<float, 8, 1> &mean,
                    Eigen::Matrix<float, 8, 8> &covariance) {
            Eigen::Matrix<float, 4, 1> mean_ret;
            Eigen::Matrix<float, 4, 4> covariance_ret;
            project(mean, covariance, mean_ret, covariance_ret);

            Eigen::Map<Eigen::MatrixXf> cov_map(covariance_ret.data(), covariance_ret.rows(), covariance_ret.cols());
            auto cov_inv = cov_map.inverse();
            auto kalman_gain = covariance * update_mat_.transpose() * cov_inv;

            Eigen::Matrix<float, 4, 1> measure;
            measure << boxah.center_x, boxah.center_y, boxah.aspect_ratio, boxah.height;
            auto innovation = measure - mean_ret;
            
            mean = mean + kalman_gain * innovation;
            covariance = covariance - kalman_gain * update_mat_ * covariance;
        }

        void initiate(const BBoxXYAH &boxah, Eigen::Matrix<float, 8, 1> &mean, 
                    Eigen::Matrix<float, 8, 8> &covariance) {
            mean << boxah.center_x, boxah.center_y, boxah.aspect_ratio,
                boxah.height, 0.0f, 0.0f, 0.0f, 0.0f;

            /** 初始状态 **/
            Eigen::Matrix<float, 8, 1> std_val;
            // std_val << 2.0f * std_weight_position_ * boxah.height,
            //            2.0f * std_weight_position_ * boxah.height,
            //            1e-2,
            //            2.0f * std_weight_position_ * boxah.height,

            //            10.0f * std_weight_velocity_ * boxah.height,
            //            10.0f * std_weight_velocity_ * boxah.height,
            //            1e-5,
            //            10.0f * std_weight_velocity_ * boxah.height;
            std_val << config_.initiate_state[0] * boxah.height,
                       config_.initiate_state[1] * boxah.height,
                       config_.initiate_state[2],
                       config_.initiate_state[3] * boxah.height,
                       config_.initiate_state[4] * boxah.height,
                       config_.initiate_state[5] * boxah.height,
                       config_.initiate_state[6],
                       config_.initiate_state[7] * boxah.height;
            covariance = Eigen::Matrix<float, 8, 8>(std_val.array().pow(2).matrix().asDiagonal());
        }

    private:
        //float std_weight_position_{1.0f / 20};
        //float std_weight_velocity_{1.0f / 160};

        Eigen::Matrix<float, 8, 8> motion_mat_;
        Eigen::Matrix<float, 4, 8> update_mat_;
        TrackerConfig config_;
    };

    class TrackObjectImpl : public TrackObject
    {
    public:
        TrackObjectImpl(const Box &box, 
                    const Eigen::Matrix<float, 8, 1> &mean,
                    const Eigen::Matrix<float, 8, 8> &covariance,
                    int id_next, int nbuckets, int max_age, int nhit, bool has_feature)
            :nbuckets_(nbuckets), max_age_(max_age), nhit_(nhit), has_feature_(has_feature)
        {
            last_position_ = box;
            covariance_    = covariance;
            mean_          = mean;
            id_            = id_next;
            state_         = State::Tentative;
            trace_.emplace_back(box);

            if(has_feature_)
                feature_bucket_.push_back(box.feature);
        }

        virtual int time_since_update() const {return time_since_update_;}
        virtual State state() const {return state_;}
        virtual Box last_position() const {return last_position_;}
        virtual int id() const {return id_;}
        virtual bool is_confirmed() const {return state_ == State::Confirmed;}

        virtual int trace_size() const{
            return trace_.size();
        }

        virtual Box& location(int time_since_update){
            static Box none;
            if(time_since_update >= trace_.size() || time_since_update < 0){
                printf("time_since_update[%d] out of range[%d]\n", time_since_update, trace_.size());
                return none;
            }
            return trace_[(int)trace_.size() - 1 - time_since_update];
        }

        Eigen::Matrix<float, 8, 1> get_mean() const {return mean_;}
        Eigen::Matrix<float, 8, 8> get_covariance() const {return covariance_;}

        virtual Box predict_box() const {
            float center_x = mean_(0, 0);
            float center_y = mean_(1, 0);
            float aspect_ratio = mean_(2, 0);
            float height = mean_(3, 0);
            float width = aspect_ratio * height;

            float left = int(center_x - width / 2);
            float top = int(center_y - height / 2);
            float right = int(center_x + width / 2);
            float bottom = int(center_y + height / 2);

            return Box(left, top, right, bottom);
        }

        void predict(KalmanFilter &km_filter) {
            km_filter.predict(mean_, covariance_);

            ++ age_;
            ++ time_since_update_;
        }

        void mark_missed() {
            if (state_ == State::Tentative || time_since_update_ > max_age_) {
                state_ = State::Deleted;
            }
        }

        void update(KalmanFilter &km_filter, const Box &box) {
            
            if(has_feature_ && box.feature.empty()){
                fprintf(stderr, "Feature is empty, ignore has_feature_ flag\n");
                has_feature_ = false;
            }

            if(has_feature_){
                if(feature_bucket_.rows < nbuckets_){
                    feature_bucket_.push_back(box.feature);
                }else{
                    box.feature.copyTo(feature_bucket_.row(feature_cursor_++));

                    if(feature_cursor_ >= nbuckets_)
                        feature_cursor_ = 0;
                }
            }

            trace_.push_back(box);
            if (trace_.size() > nbuckets_) {
                trace_.pop_front();
            }

            km_filter.update(box, mean_, covariance_);
            last_position_ = box;
            ++ hits_;
            time_since_update_ = 0;

            if (state_ == State::Tentative && hits_ >= nhit_) {
                state_ = State::Confirmed;
            }
        }

        virtual const cv::Mat& feature_bucket() const override{
            return feature_bucket_;
        }

        virtual std::vector<cv::Point> trace_line() const {
            std::vector<cv::Point> line;
            const int Count = trace_.size();
            const int Smooth = 5;
            for (int i = 0; i < Count; ++i) {
                int begin = std::max<int>(0, i - Smooth / 2);
                int end = std::min<int>(i + Smooth / 2 + 1, Count);
                int x = 0;
                int y = 0;
                for (int j = begin; j < end; ++j) {
                    x += trace_[j].center().x;
                    y += trace_[j].bottom;
                }
                line.push_back(cv::Point(x / (end - begin), y  / (end - begin)));
            }
            return line;
        }

    private:
        int time_since_update_{0};
        State state_{State::Tentative};
        int age_{1};
        int hits_{1};
        int id_;
        int feature_cursor_ = 0;
        std::deque<Box> trace_;
        cv::Mat feature_bucket_;
        bool has_feature_ = false;

        int nbuckets_ = 100;
        int max_age_ = 100;
        int nhit_ = 3;

        Box last_position_;
        Eigen::Matrix<float, 8, 1> mean_;
        Eigen::Matrix<float, 8, 8> covariance_;
    };

    class TrackerImpl : public Tracker
    {
    public:
        TrackerImpl(const TrackerConfig& config)
        :kalman_(config), 
        distance_threshold_(config.distance_threshold), 
        nbuckets_(config.nbuckets), 
        max_age_(config.max_age), 
        nhit_(config.nhit), 
        has_feature_(config.has_feature) {
        }

        virtual ~TrackerImpl() {
        }

        virtual std::vector<TrackObject *> get_objects() {
            std::vector<TrackObject *> objects_ptr;
            for (TrackObjectImpl &obj : objects_) {
                objects_ptr.push_back(&obj);
            }
            return objects_ptr;
        }

        void predict() {
            for (auto &obj : objects_) {
                obj.predict(kalman_);
            }
        }

        void update(const BBoxes& boxes) {

            predict();

            int level_max = max_age_;
            State states[2] = {State::Confirmed, State::Tentative};
            std::vector<int> unmatched_boxes_index, unmatched_objects_index;
            for (int i = 0; i < boxes.size(); ++i) {
                unmatched_boxes_index.push_back(i);
            }
            for (int i = 0; i < objects_.size(); ++i) {
                unmatched_objects_index.push_back(i);
            }

            std::vector<int> match_boxes_index;
            std::vector<int> match_objects_index;
            for (auto state : states) {
                for (int level = 0; level < level_max; ++level) {
                    if (unmatched_boxes_index.size() == 0 || unmatched_objects_index.size() == 0) {
                        break;
                    }
                    std::vector<int> objects_index;
                    for (auto index : unmatched_objects_index) {
                        if (objects_[index].time_since_update() == level + 1 &&
                            objects_[index].state() == state) {
                            objects_index.push_back(index);
                        }
                    }
                    if (objects_index.size() == 0) {
                        continue;
                    }

                    // match
                    match_boxes_index.clear();
                    match_objects_index.clear();
                    this->match(objects_index, unmatched_boxes_index, boxes, 
                                match_boxes_index, match_objects_index);

                    // unmatch boxes index
                    std::set<int> match_boxes_set(match_boxes_index.begin(), match_boxes_index.end());
                    std::set<int> unmatched_boxes_set(unmatched_boxes_index.begin(), unmatched_boxes_index.end());
                    unmatched_boxes_index.clear();
                    std::set_symmetric_difference(
                        match_boxes_set.begin(), match_boxes_set.end(),
                        unmatched_boxes_set.begin(), unmatched_boxes_set.end(),
                        std::back_inserter(unmatched_boxes_index)
                    );

                    // unmatched_objects_index
                    std::set<int> unmatched_objects_set(unmatched_objects_index.begin(), unmatched_objects_index.end());
                    std::set<int> match_objects_set(match_objects_index.begin(), match_objects_index.end());
                    unmatched_objects_index.clear();
                    std::set_symmetric_difference(
                        unmatched_objects_set.begin(), unmatched_objects_set.end(),
                        match_objects_set.begin(), match_objects_set.end(),
                        std::back_inserter(unmatched_objects_index)
                    );

                    // update
                    int count = std::min<int>(match_objects_index.size(), match_boxes_index.size());
                    for (int i = 0; i < count; ++i) {
                        objects_[match_objects_index[i]].update(kalman_, boxes[match_boxes_index[i]]);
                    }
                }
            }

            for (auto index : unmatched_objects_index) {
                objects_[index].mark_missed();
            }
            for (auto index : unmatched_boxes_index) {
                this->new_object(boxes[index]);
            }
            std::vector<TrackObjectImpl> objects_tmp;
            std::copy_if(objects_.begin(), objects_.end(), std::back_inserter(objects_tmp),
                        [](const TrackObject &obj){return obj.state() != State::Deleted;}
                        );
            objects_ = objects_tmp;
        }

        void match(const std::vector<int> &objects_index, 
                const std::vector<int> &boxes_index, 
                const std::vector<Box> &boxes,
                std::vector<int> &match_boxes_index,
                std::vector<int> &match_objects_index) {
            std::vector<std::vector<double>> cost_matrix_data;
            for (auto obj_idx : objects_index) {
                std::vector<double> cost_matrix_item;
                for (auto box_idx : boxes_index) {
                    auto &TrackObject = objects_[obj_idx];
                    auto &box = boxes[box_idx];
                    BBoxXYAH boxah(box);

                    auto maha_distance = kalman_.ma_distance(
                        TrackObject.get_mean(), TrackObject.get_covariance(),
                        boxah, false
                    );

                    double cost_data = 0;
                    if (maha_distance > chi2inv95_2[3]) {
                        cost_data = 1e5;
                    }
                    else {
                        if(has_feature_){
                            cv::Mat scores   = TrackObject.feature_bucket() * box.feature.t();
                            double max_score = 0;
                            cv::minMaxLoc(scores, nullptr, &max_score);
                            cost_data = 1 - max_score;
                        }else{
                            cost_data = distance(TrackObject.last_position(), box);
                        }
                    }
                    cost_matrix_item.push_back(cost_data);
                }
                cost_matrix_data.push_back(cost_matrix_item);
            }

            HungarianAlgorithm HungAlgo;
            std::vector<int> assignment;
            double cost = HungAlgo.Solve(cost_matrix_data, assignment);
        
            for (int i = 0; i < assignment.size(); ++i) {
                if (assignment[i] < 0) {
                    continue;
                }
                int obj_index = objects_index[i];
                int box_index = boxes_index[assignment[i]];
                if (cost_matrix_data[i][assignment[i]] < distance_threshold_) {
                    match_boxes_index.push_back(box_index);
                    match_objects_index.push_back(obj_index);
                }
            }
        }

        void new_object(const Box &box) {
            Eigen::Matrix<float, 8, 1> mean;
            Eigen::Matrix<float, 8, 8> covariance;
            kalman_.initiate(BBoxXYAH(box), mean, covariance);

            objects_.emplace_back(box, mean, covariance, id_next_, nbuckets_, max_age_, nhit_, has_feature_);
            ++ id_next_;
        }

    private:
        int id_next_{1};
        std::vector<TrackObjectImpl> objects_;
        KalmanFilter kalman_;
        float distance_threshold_ = 0;
        int nbuckets_ = 100;
        int max_age_ = 100;
        int nhit_ = 3;
        bool has_feature_ = false;
    };

    std::shared_ptr<Tracker> create_tracker(const TrackerConfig& config) {

        if(config.has_feature && config.nbuckets < 1 || config.max_age < 1 || config.nhit < 1){
            printf("Invalid argument has_feature = %s, nbuckets = %d, max_age = %d, nhit = %d\n", config.has_feature ? "True":"False", config.nbuckets, config.max_age, config.nhit);
            return nullptr;
        }

        std::shared_ptr<TrackerImpl> tracker_ptr(new TrackerImpl(
            config
        ));
        return tracker_ptr;
    }
};
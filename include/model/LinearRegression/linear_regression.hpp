#ifndef AMT_ML_MODEL_LINEAR_REGRESSION_HPP
#define AMT_ML_MODEL_LINEAR_REGRESSION_HPP

#include <model/LinearRegression/optimizer.hpp>
#include <dataframe.hpp>

namespace amt::regression{

    struct LinearRegression{
        LinearRegression() = default;
        LinearRegression(LinearRegression const& other) = default;
        LinearRegression(LinearRegression && other) = default;
        LinearRegression& operator=(LinearRegression const& other) = default;
        LinearRegression& operator=(LinearRegression && other) = default;
        ~LinearRegression() = default;

        template<typename Optimizer = regression::default_opt>
        requires regression::is_default_opt_v<Optimizer> || regression::is_gradient_descent_v<Optimizer>
        LinearRegression(FrameViewOrFrame auto const& x, 
            FrameViewOrFrame auto const& y,
            double lambda = 0.0,
            Optimizer opt = {},
            bool intercept = true
        ){
            if( x.rows() != y.rows() ){
                throw std::runtime_error(
                    "amt::LinearRegression(FrameViewOrFrame auto const&, FrameViewOrFrame auto const&, bool) : "
                    "Rows of the Dependent Variable(Y) should be equal to Rows of the Independent Variable(X)"
                );
            }
            
            if( y.cols() != 1u ){
                throw std::runtime_error(
                    "amt::LinearRegression(FrameViewOrFrame auto const&, FrameViewOrFrame auto const&, bool) : "
                    "Cols of the Dependent Variable(Y) should be one"
                );
            }
            arma::Mat<double> X(x.rows(),x.cols() + static_cast<std::size_t>(intercept)),Y(y.rows(),1u);
            
            assignX(X,x);

            auto i = 0ul;
            for(auto const& el : y[0]){
                Y(i++,0) = el.template as<double>();
            }

            opt(m_data,X,Y,lambda);
        }

        void assignX(arma::Mat<double>& X, FrameViewOrFrame auto& x){
            for(auto i = 0ul; i < x.rows(); ++i){
                for(auto j = 0ul; j < x.cols(); ++j){
                    X(i,j + 1ul) = x[j][i].template as<double>();
                }
            }

            for(auto j = 0ul; j < x.rows(); ++j){
                X(j,0ul) = 1.0;
            }
        }

        amt::frame<> predict(FrameViewOrFrame auto& x){
            amt::frame<> res(1ul, x.rows());
            arma::Mat<double> X(x.rows(), x.cols() + 1);
            assignX(X,x);
            arma::Mat<double> p = X * m_data;
            for(auto i = 0ul; i < x.rows(); ++i){
                res[0][i] = p(i,0);
            }
            res.set_name(std::vector<std::string>{"Prediction"});
            return res;
        }

        constexpr double beta(std::size_t k) const{
            return m_data(k,0);
        }

        series<> beta() const{
            series<> temp(m_data.n_rows, 0.0, "Beta or Coefficient of X");
            for(auto i = 0u; i < temp.size(); ++i) temp[i] = m_data(i,0);
            return temp;
        }

    private:
        arma::Mat<double> m_data;
    };

} // amt::regression


#endif

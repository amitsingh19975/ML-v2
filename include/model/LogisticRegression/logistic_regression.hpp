#ifndef AMT_ML_MODEL_LOGISTIC_REGRESSION_HPP
#define AMT_ML_MODEL_LOGISTIC_REGRESSION_HPP

#include <model/LogisticRegression/optimizer.hpp>
#include <dataframe.hpp>

namespace amt{

    struct LogisticRegression{
        LogisticRegression() = default;
        LogisticRegression(LogisticRegression const& other) = default;
        LogisticRegression(LogisticRegression && other) = default;
        LogisticRegression& operator=(LogisticRegression const& other) = default;
        LogisticRegression& operator=(LogisticRegression && other) = default;
        ~LogisticRegression() = default;

        template<typename Optimizer = logistic_regression::gradient_descent>
        requires logistic_regression::is_default_opt_v<Optimizer> || logistic_regression::is_gradient_descent_v<Optimizer>
        LogisticRegression(FrameViewOrFrame auto const& x, 
            FrameViewOrFrame auto const& y,
            Optimizer opt = {},
            bool intercept = true
        ){
            if( x.rows() != y.rows() ){
                throw std::runtime_error(
                    "amt::LogisticRegression(FrameViewOrFrame auto const&, FrameViewOrFrame auto const&, bool) : "
                    "Rows of the Dependent Variable(Y) should be equal to Rows of the Independent Variable(X)"
                );
            }
            
            if( y.cols() != 1u ){
                throw std::runtime_error(
                    "amt::LogisticRegression(FrameViewOrFrame auto const&, FrameViewOrFrame auto const&, bool) : "
                    "Cols of the Dependent Variable(Y) should be one"
                );
            }
            arma::Mat<double> X(x.rows(),x.cols() + static_cast<std::size_t>(intercept)),Y(y.rows(),1u);
            
            assignX(X,x);

            auto i = 0ul;
            for(auto const& el : y[0]){
                Y(i++,0) = el.template as<double>();
            }

            opt(m_data,X,Y);
        }

        void assignX(arma::Mat<double>& X, FrameViewOrFrame auto& x) const{
            for(auto i = 0ul; i < x.rows(); ++i){
                for(auto j = 0ul; j < x.cols(); ++j){
                    X(i,j + 1ul) = x[j][i].template as<double>();
                }
            }

            for(auto j = 0ul; j < x.rows(); ++j){
                X(j,0ul) = 1.0;
            }
        }

        amt::frame<> predict_prob(FrameViewOrFrame auto& x) const{
            amt::frame<> res(1ul, x.rows());
            arma::Mat<double> X(x.rows(), x.cols() + 1);
            assignX(X,x);
            logistic_regression::gradient_descent grad;
            arma::Mat<double> beta = X * m_data;
            arma::Mat<double> p = grad.apply_fn(beta, grad.fn);
            for(auto i = 0ul; i < x.rows(); ++i){
                res[0][i] = p(i,0);
            }
            res.set_name(std::vector<std::string>{"Prediction"});
            return res;
        }

        amt::frame<> predict(FrameViewOrFrame auto& x, double threshold = 0.5) const{
            amt::frame<> res(1ul, x.rows());
            arma::Mat<double> X(x.rows(), x.cols() + 1);
            assignX(X,x);
            logistic_regression::gradient_descent grad;
            arma::Mat<double> beta = X * m_data;
            arma::Mat<double> p = grad.apply_fn(beta, grad.fn);
            for(auto i = 0ul; i < x.rows(); ++i){
                res[0][i] = ( p(i,0) < threshold ? 0.0 : 1.0);
            }
            res.set_name(std::vector<std::string>{"Prediction"});
            return res;
        }

        auto predict_prob(arma::Mat<double> const& X, double threshold = 0.5) const{
            logistic_regression::gradient_descent grad;
            arma::Mat<double> beta = X * m_data;
            arma::Mat<double> p = grad.apply_fn(beta, [&threshold,&grad](double v){
                return grad.fn(v) < threshold ? 0 : 1;
            });
            return p;
        }

        auto predict(arma::Mat<double> const& X) const{
            logistic_regression::gradient_descent grad;
            arma::Mat<double> beta = X * m_data;
            arma::Mat<double> p = grad.apply_fn(beta, grad.fn);
            return p;
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

} // amt


#endif

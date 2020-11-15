#ifndef AMT_ML_MODEL_LOGISTIC_REGRESSION_HPP
#define AMT_ML_MODEL_LOGISTIC_REGRESSION_HPP

#include <model/LogisticRegression/optimizer.hpp>
#include <dataframe.hpp>

namespace amt::classification::detail{

    struct LogisticRegression{
        LogisticRegression() = default;
        LogisticRegression(LogisticRegression const& other) = default;
        LogisticRegression(LogisticRegression && other) = default;
        LogisticRegression& operator=(LogisticRegression const& other) = default;
        LogisticRegression& operator=(LogisticRegression && other) = default;
        ~LogisticRegression() = default;

        LogisticRegression(Frame auto const& x, 
            Frame auto const& y,
            gradient_descent opt = {},
            bool intercept = true
        )
            : grad(std::move(opt))
            , intercept(intercept)
        {
            if( x.rows() != y.rows() ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Rows of the Dependent Variable(Y) should be equal to Rows of the Independent Variable(X)"
                );
            }
            
            if( y.cols() != 1u ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Cols of the Dependent Variable(Y) should be one"
                );
            }
            arma::Mat<double> X(x.rows(),x.cols() + static_cast<std::size_t>(intercept)),Y(y.rows(),1u);

            assignX(X,x);

            auto i = 0ul;
            for(auto const& el : y[0]){
                Y(i++,0) = get<double>(el);
            }
            
            grad(m_data,X,Y);
        }

        void assignX(arma::Mat<double>& X, Frame auto& x) const{

            for(auto i = 0ul; i < x.rows(); ++i){
                for(auto j = 0ul; j < x.cols(); ++j){
                    X(i,j + static_cast<std::size_t>(intercept)) = get<double>(x[j][i]);
                }
            }

            if(intercept){
                for(auto j = 0ul; j < x.rows(); ++j){
                    X(j,0ul) = 1.0;
                }
            }
        }

        amt::frame predict_prob(Frame auto& x) const{
            
            amt::frame res(1u, x.rows());
            
            arma::Mat<double> X(x.rows(), x.cols() + static_cast<std::size_t>(intercept));
            assignX(X,x);
            std::vector<std::string> names;

            arma::Mat<double> beta = X * m_data;
            for(auto j = 0ul; j < x.rows(); ++j){
                res[0][j] = grad.fn( beta(j,0) );
            }
            res[0].name() = "Prediction";
            return res;
        }

        amt::frame predict(Frame auto& x, double threshold = 0.5) const{
            auto res = predict_prob(x);
            amt::transform(res,amt::tag::inplace,[threshold](double v){
                return static_cast<double>(v > threshold);
            });
            return res;
        }

        auto predict_prob(arma::Mat<double> const& X, double threshold = 0.5) const{
            arma::Mat<double> beta = X * m_data[0];
            arma::Mat<double> p = grad.apply_fn(beta, [&threshold,this](double v){
                return static_cast<double>(grad.fn(v) > threshold);
            });
            return p;
        }

        auto predict(arma::Mat<double> const& X) const{
            arma::Mat<double> beta = X * m_data[0];
            arma::Mat<double> p = grad.apply_fn(beta, grad.fn);
            return p;
        }

        constexpr double beta(std::size_t k) const{
            return m_data(k,0);
        }

        auto beta() const{
            series s("Beta ( Coefficients of X )", m_data.n_rows, 0.0, dtype<double>());
            for(auto i = 0u; i < s.size(); ++i) s[i] = m_data(i,0);
            return s;
        }

    private:
        arma::Mat<double> m_data;
        gradient_descent grad;
        bool intercept{true};
    };

    struct LogisticRegressionOVR{
        LogisticRegressionOVR() = default;
        LogisticRegressionOVR(LogisticRegressionOVR const& other) = default;
        LogisticRegressionOVR(LogisticRegressionOVR && other) = default;
        LogisticRegressionOVR& operator=(LogisticRegressionOVR const& other) = default;
        LogisticRegressionOVR& operator=(LogisticRegressionOVR && other) = default;
        ~LogisticRegressionOVR() = default;

        LogisticRegressionOVR(Frame auto const& x, 
            Frame auto const& y,
            gradient_descent opt = {},
            bool intercept = true
        )
            : grad(std::move(opt))
            , intercept(intercept)
        {
            if( x.rows() != y.rows() ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Rows of the Dependent Variable(Y) should be equal to Rows of the Independent Variable(X)"
                );
            }
            
            if( y.cols() != 1u ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Cols of the Dependent Variable(Y) should be one"
                );
            }
            arma::Mat<double> X(x.rows(),x.cols() + static_cast<std::size_t>(intercept)),Y(y.rows(),1u);

            assignX(X,x);

            auto i = 0ul;
            for(auto const& el : y[0]){
                Y(i++,0) = get<double>(el);
            }

            auto max = static_cast<std::size_t>(Y.max());
            auto sz = max + 1ul;

            m_data.resize(sz);

            for(i = 0u; i < sz; ++i){
                auto y_temp = Y;
                for(auto j = 0u; j < y_temp.n_elem; ++j){
                    y_temp[j] = static_cast<double>( static_cast<std::size_t>(y_temp[j]) == i ? 1 : 0 );
                } 
                
                grad(m_data[i],X,y_temp);
            }
        }

        LogisticRegressionOVR(Frame auto const& x, 
            Frame auto const& y,
            std::size_t labels,
            gradient_descent opt = {},
            bool intercept = true
        )
            : m_data(labels)
            , grad(std::move(opt))
            , intercept(intercept)
        {
            if( x.rows() != y.rows() ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Rows of the Dependent Variable(Y) should be equal to Rows of the Independent Variable(X)"
                );
            }
            
            if( y.cols() != 1u ){
                throw std::runtime_error(
                    "amt::LogisticRegression(Frame auto const&, Frame auto const&, bool) : "
                    "Cols of the Dependent Variable(Y) should be one"
                );
            }
            arma::Mat<double> X(x.rows(),x.cols() + static_cast<std::size_t>(intercept)),Y(y.rows(),1u);

            assignX(X,x);

            auto i = 0ul;
            for(auto const& el : y[0]){
                Y(i++,0) = get<double>(el);
            }

            for(i = 0u; i < labels; ++i){
                auto y_temp = Y;
                for(auto j = 0u; j < y_temp.n_elem; ++j){
                    y_temp[j] = static_cast<double>( static_cast<std::size_t>(y_temp[j]) == i ? 1 : 0 );
                } 
                
                grad(m_data[i],X,y_temp);
            }
        }

        void assignX(arma::Mat<double>& X, Frame auto& x) const{

            for(auto i = 0ul; i < x.rows(); ++i){
                for(auto j = 0ul; j < x.cols(); ++j){
                    X(i,j + static_cast<std::size_t>(intercept)) = get<double>(x[j][i]);
                }
            }

            if(intercept){
                for(auto j = 0ul; j < x.rows(); ++j){
                    X(j,0ul) = 1.0;
                }
            }
        }

        amt::frame predict_prob(Frame auto& x) const{
            auto cols = m_data.size();
            amt::frame res(cols, x.rows());
            res.dtype(dtype<double>());
            arma::Mat<double> X(x.rows(), x.cols() + static_cast<std::size_t>(intercept));
            assignX(X,x);
            std::vector<std::string> names;
            for(auto i = 0u; i < cols; ++i){
                arma::Mat<double> beta = X * m_data[i];
                for(auto j = 0ul; j < x.rows(); ++j){
                    res[i][j] = grad.fn( beta(j,0) );
                }
                names.push_back( std::string("Prediction ") + std::to_string(i) );
            }
            res.set_names(names);
            return res;
        }

        amt::frame predict(Frame auto& x) const{
            auto res = predict_prob(x);
            for(auto i = 0u; i < res.rows(); ++i){
                auto idx = 0ul;
                auto max = get<double>(res[0][i]);
                for(auto j = 1ul; j < res.cols(); ++j){
                    auto const& el = get<double>(res[j][i]);
                    if( max < el ){
                        max = el;
                        idx = j;
                    }
                }
                res[0][i] = static_cast<double>(idx);
            }
            res.resize(1u);
            res[0].name().pop_back();
            res[0].name().pop_back();
            return res;
        }

        constexpr double beta(std::size_t k, std::size_t m = 0u) const{
            return m_data[m](k,0);
        }

        auto beta() const{
            frame temp;
            for(auto k = 0u; k < m_data.size(); ++k){
                series s(std::to_string(k), m_data[k].n_rows, 0.0);
                for(auto i = 0u; i < s.size(); ++i) s[i] = m_data[k](i,0);
                temp.col_push_back(std::move(s));
            }
            return temp;
        }

    private:
        std::vector<arma::Mat<double>> m_data;
        gradient_descent grad;
        bool intercept{true};
    };

} // amt::classification::detail

namespace amt::classification{
    struct OVR{};
    template< typename T = void >
    using LogisticRegression = std::conditional_t<
        std::is_same_v<OVR,T>,
        detail::LogisticRegressionOVR,
        detail::LogisticRegression
    >;
} // amt::classification

#endif

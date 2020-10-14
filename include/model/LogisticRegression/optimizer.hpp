#ifndef AMT_ML_MODEL_LINEAR_REGRESSION_OPTIMIZER_HPP
#define AMT_ML_MODEL_LINEAR_REGRESSION_OPTIMIZER_HPP

#include <armadillo>
#include <functional>
#include <cmath>

namespace amt::logistic_regression{
    
    struct default_opt{

    //     auto operator()(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y, double lm) const {
    //         // x_t = x^T
    //         auto x_t = x.t();
            
    //         // xt_y = x^T * y
    //         auto xt_y = x_t * y;

    //         if( lm == 0 ){
    //             // xt_x = x^T * x
    //             auto xt_x = x_t * x;

    //             // (X * X^T) * B = X^T * Y
    //             beta = arma::solve(xt_x,xt_y,arma::solve_opts::fast);
    //         }else{
                
    //             arma::Mat<double> l_I = arma::eye(x.n_cols, x.n_cols) * lm;

    //             // xt_x = x^T * x + l_I
    //             auto xt_x_lI = ( x_t * x ) + l_I;

    //             // (X * X^T) * B = X^T * Y
    //             beta = arma::solve(xt_x_lI,xt_y,arma::solve_opts::fast);
    //         }

    //     }

    };
    
    struct gradient_descent{

        gradient_descent() = default;
        gradient_descent(gradient_descent const&  other) = default;
        gradient_descent(gradient_descent &&  other) = default;
        gradient_descent& operator=(gradient_descent const&  other) = default;
        gradient_descent& operator=(gradient_descent &&  other) = default;
        ~gradient_descent() = default;
        
        template<typename Fn>
        gradient_descent(double alpha, std::size_t iter, Fn&& fn)
            : alpha(alpha)
            , iteration(iter)
            , fn(std::move(fn))
        {}

        gradient_descent(double alpha, std::size_t iter = 300000)
            : alpha(alpha)
            , iteration(iter)
        {}

        auto operator()(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y) const {
            beta.resize(x.n_cols, 1ul);
            beta.randu();
            eval(beta,x,y);
        }

        void eval(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y) const{
            auto m = alpha / static_cast<double>(x.n_rows);
            auto x_t = x.t();
            for( auto i = 0u; i < iteration; ++i ){
                auto p = ( x * beta );
                auto h = apply_fn(p,fn);
                arma::Mat<double> grad = ( x_t * (h - y) );
                beta -= ( m * grad );
            }
        }

        arma::Mat<double> apply_fn(arma::Mat<double> const& mat, std::function<double(double)> const& f) const{
            arma::Mat<double> res(mat.n_rows, mat.n_cols);
            for(auto i = 0u; i < mat.n_rows; ++i)
                for(auto j = 0u; j < mat.n_cols; ++j)
                    res(i,j) = f(mat(i,j));
            return res;
        }


        double alpha{0.1};
        std::size_t iteration{300000};
        std::function<double(double)> fn = [](double el){
            return 1.0 / ( 1.0 + std::exp(-el) );
        };
        std::function<double(double)> diff_fn = [](double el){
            auto temp = std::exp(-el);
            return temp / ( 1.0 + temp );
        };
    };

    template<typename T>
    inline static constexpr bool is_default_opt_v = std::is_same_v<T,default_opt>;

    template<typename T>
    inline static constexpr bool is_gradient_descent_v = std::is_same_v<T,gradient_descent>;

} // namespace amt::logistic_regression


#endif

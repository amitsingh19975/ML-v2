#ifndef AMT_ML_MODEL_LINEAR_REGRESSION_OPTIMIZER_HPP
#define AMT_ML_MODEL_LINEAR_REGRESSION_OPTIMIZER_HPP

#include <armadillo>
#include <functional>

namespace amt::regression{
    
    struct default_opt{

        auto operator()(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y, double lm) const {
            // x_t = x^T
            auto x_t = x.t();
            
            // xt_y = x^T * y
            auto xt_y = x_t * y;

            if( lm == 0 ){
                // xt_x = x^T * x
                auto xt_x = x_t * x;

                // (X * X^T) * B = X^T * Y
                beta = arma::solve(xt_x,xt_y,arma::solve_opts::fast);
            }else{
                
                arma::Mat<double> l_I = arma::eye(x.n_cols, x.n_cols) * lm;

                // xt_x = x^T * x + l_I
                auto xt_x_lI = ( x_t * x ) + l_I;

                // (X * X^T) * B = X^T * Y
                beta = arma::solve(xt_x_lI,xt_y,arma::solve_opts::fast);
            }

        }

    };
    
    struct gradient_descent{

        constexpr gradient_descent() noexcept = default;
        constexpr gradient_descent(gradient_descent const&  other) noexcept = default;
        constexpr gradient_descent(gradient_descent &&  other) noexcept = default;
        constexpr gradient_descent& operator=(gradient_descent const&  other) noexcept = default;
        constexpr gradient_descent& operator=(gradient_descent &&  other) noexcept = default;
        ~gradient_descent() = default;
        
        constexpr gradient_descent(double alpha, std::size_t iter)
            : alpha(alpha)
            , iteration(iter)
        {}

        auto operator()(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y, double lm) const {
            beta.resize(x.n_cols, 1ul);
            beta.randu();
            if( lm == 0.0 )
                eval(beta,x,y,lm);
            else
                eval(beta,x,y);
        }

        // double compute_cost(arma::Mat<double> const& beta, arma::Mat<double> const& x, arma::Mat<double> const& y){
        //     auto temp = ( x * beta ) - y;
        //     auto m = 2 * x.n_rows;
        //     auto J = ( temp * temp.t() ) / static_cast<double>(m);
        //     return J[0];
        // }

        void eval(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y, double lm) const{
            auto m = alpha / static_cast<double>(x.n_rows);
            auto reg_coef = m * lm;

            auto x_t = x.t();
            for( auto i = 0u; i < iteration; ++i ){
                auto p = ( x * beta ) - y;
                auto temp = m * (x_t * p);
                beta -= ( temp + ( reg_coef * beta ) );
            }
        }

        void eval(arma::Mat<double>& beta, arma::Mat<double> const& x, arma::Mat<double> const& y) const{
            auto m = alpha / static_cast<double>(x.n_rows);
            auto x_t = x.t();
            for( auto i = 0u; i < iteration; ++i ){
                auto p = ( x * beta ) - y;
                auto temp = m * (x_t * p);
                beta -= temp;
            }
        }


        double alpha{0.01};
        std::size_t iteration{1500};
    };

    template<typename T>
    inline static constexpr bool is_default_opt_v = std::is_same_v<T,default_opt>;

    template<typename T>
    inline static constexpr bool is_gradient_descent_v = std::is_same_v<T,gradient_descent>;

} // namespace amt::regression


#endif

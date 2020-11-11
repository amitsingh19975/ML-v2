#if !defined(AMT_KERNEL_GAUSSIAN_HPP)
#define AMT_KERNEL_GAUSSIAN_HPP

#define _USE_MATH_DEFINES

#include <dataframe.hpp>
#include <cmath>

namespace amt::kernel{

    struct Gaussian{

        double operator()(double val, double meank, double vark) const noexcept{
            // z = (v - mean)^2 / (2 * var)
            auto z = val - meank;
            z = ( z * z ) / (2. * vark);

            // exp = e ^ -z
            auto exp = std::exp(-z);

            // 2 * pi * var
            auto den =  std::sqrt( 2.0 * M_PI * vark );

            return exp / den;
        }

        auto& operator()(Series auto& s, tag::inplace_t) const noexcept{
            auto m = mean<>(s);
            auto v = var<>(s);

            for(auto& el : s){
                double val = el;
                el = this->operator()(val,m,v);
            }
            return s;
        }

        auto operator()(Series auto const& s) const noexcept{
            auto m = mean<>(s);
            auto v = var<>(s);
            auto temp = s;

            for(auto& el : temp){
                double val = el;
                el = this->operator()(val,m,v);
            }
            return temp;
        }

    };

} // namespace amt::kernel

#endif // AMT_KERNEL_GAUSSIAN_HPP

#if !defined(AMT_METRICS_REGRESSION_HPP)
#define AMT_METRICS_REGRESSION_HPP

#include <dataframe.hpp>
#include <sstream>
#include <ostream>

namespace amt::regression{

    struct metrics{
        double max_err{};
        double mean_err{};
        double mean_abs_err{};
        double mean_sq_err{};
        double r2_score{};

        friend std::ostream& operator<<(std::ostream& os, metrics const& m){
            os<<std::quoted("Max Error")<<"              "<<std::to_string(m.max_err)<<'\n';
            os<<std::quoted("Mean Error")<<"             "<<std::to_string(m.mean_err)<<'\n';
            os<<std::quoted("Mean Absolute Error")<<"    "<<std::to_string(m.mean_abs_err)<<'\n';
            os<<std::quoted("Mean Squared Error")<<"     "<<std::to_string(m.mean_sq_err)<<'\n';
            os<<std::quoted("R^2 Scrore")<<"             "<<std::to_string(m.r2_score)<<'\n';
            return os;
        }
    };

    metrics calculate_metrics(Series auto const& y_pred, Series auto const& y_true){
        if( !is_floating_point(y_pred) || !is_floating_point(y_true) ){
            throw std::runtime_error("predicted value and true value must be floating point");
        }
        auto diff = y_true - y_pred;
        metrics m;
        auto sz = static_cast<double>(diff.size());
        m.max_err = max<>(diff);
        m.mean_err = mean<>(diff);

        m.mean_abs_err = reduce_col(diff,0.0,[](double const& p, double const& val){
            return p + std::abs(val);
        });
        m.mean_abs_err /= sz;

        m.mean_sq_err = reduce_col(diff,0.0,[](double const& p, double const& val){
            return p + val * val;
        });
        m.mean_sq_err /= sz;

        auto y_true_mean = mean<>(y_true);

        auto ss_total = reduce_col(y_true, 0., [y_true_mean](double const& p, double const& val){
            auto temp = val - y_true_mean;
            return p + temp * temp;
        });

        m.r2_score = 1. - (m.mean_sq_err / ss_total);

        return m; 
    }

    metrics calculate_metrics(Frame auto const& y_pred, Frame auto const& y_true){
        return calculate_metrics(y_pred[0], y_true[0]);
    }

} // namespace amt::regression

#endif // AMT_METRICS_REGRESSION_HPP

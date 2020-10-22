#if !defined(AMT_MODEL_GAUSSIAN_NAIVE_BAYES_HPP)
#define AMT_MODEL_GAUSSIAN_NAIVE_BAYES_HPP

#include <dataframe.hpp>
#include <kernel/gaussian.hpp>
#include <array>

namespace amt::classification{

    auto split_based_on_labels(FrameViewOrFrame auto const& x, FrameViewOrFrame auto const& y, std::size_t labels){
        using frame_type = std::decay_t< decltype(x) >;

        auto cols = x.cols();
        auto rows = x.rows();

        std::vector<frame_type> ret(labels, frame_type(cols));

        for(auto i = 0u; i < cols; ++i){
            for(auto j = 0u; j < rows; ++j){
                auto pos = static_cast<std::size_t>(y[0][j].template as<double>());
                auto& f = ret[pos];
                f[i].push_back(x[i][j]);
            }
        }

        for(auto& el : ret){
            el.set_name(x.names_to_vector());
        }

        return ret;
    }

    struct GaussianNB{
        
        using pair_type = std::pair<double,double>;
        using column_type = std::vector< pair_type >;
        using summary_type = std::vector< column_type >;

        GaussianNB() = default;
        GaussianNB(GaussianNB const& other) = default;
        GaussianNB(GaussianNB && other) = default;
        GaussianNB& operator=(GaussianNB const& other) = default;
        GaussianNB& operator=(GaussianNB && other) = default;
        ~GaussianNB() = default;

        GaussianNB(FrameViewOrFrame auto const& x, FrameViewOrFrame auto const& y){
            auto sz = static_cast<std::size_t>(max(y[0])) + 1;
            auto cols = x.cols();
            auto rows = x.rows();

            m_summaries.assign(sz, column_type(cols));
            m_labels_prob.assign(sz, 0.);

            auto vec_of_series = split_based_on_labels(x,y,sz);

            for(auto i = 0u; i < sz; ++i){
                m_labels_prob[i] = static_cast<double>(vec_of_series[i].rows()) / static_cast<double>(rows);
                for(auto j = 0u; j < cols; ++j){
                    auto m = mean(vec_of_series[i][j]);
                    auto v = svar(vec_of_series[i][j]);
                    m_summaries[i][j] = {m,v};
                }
            }
        }

        auto predict_prob(FrameViewOrFrame auto const& x){
            using frame_type = std::decay_t< decltype(x) >;
            auto sz = m_labels_prob.size();
            frame_type ret(sz, x.rows());

            for(auto k = 0u; k < sz; ++k){
                auto lp = m_labels_prob[k];
                for(auto j = 0u; j < x.rows(); ++j){
                    double p = lp;
                    for(auto i = 0u; i < x.cols(); ++i){
                        double val = x[i][j];
                        auto g = m_kernel(val, m_summaries[k][i].first, m_summaries[k][i].second);
                        p *= g;
                    }
                    ret[k][j] = p;
                }
            }
            return ret;
        }

        auto predict(FrameViewOrFrame auto const& x){
            using frame_type = std::decay_t< decltype(x) >;
            auto prob = predict_prob(x);
            frame_type ret(1u, x.rows());
            ret[0].name() = "Predicted Value";

            for(auto j = 0u; j < prob.rows(); ++j){
                double m = 0.;
                std::size_t idx{};
                for(auto i = 0u; i < prob.cols(); ++i){
                    double val = prob[i][j];
                    if(m < val){
                        m = val;
                        idx = i;
                    }
                }
                ret[0][j] = static_cast<double>(idx);
            }
            return ret;
        }

    private:
        summary_type m_summaries;
        std::vector<double> m_labels_prob;
        kernel::Gaussian m_kernel;
    };

} // namespace amt::classification

#endif // AMT_MODEL_GAUSSIAN_NAIVE_BAYES_HPP

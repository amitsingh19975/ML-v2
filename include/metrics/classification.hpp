#if !defined(AMT_METRICS_CLASSIFICATION_HPP)
#define AMT_METRICS_CLASSIFICATION_HPP

#include <dataframe.hpp>
#include <sstream>
#include <ostream>

namespace amt::classification{

    struct metrics{

        using base_type = std::vector<double>;
        base_type accuracy{};
        base_type balanced_accuracy{};
        base_type F1_score{};
        base_type matthews_corr_coef{};
        base_type fowlkes_mallows_idx{};
        base_type informedness{};
        base_type markedness{};

        std::vector<std::string> labels;

        void set_labels(std::vector<std::string> l){
            labels = std::move(l);
        }

        constexpr std::size_t total() const noexcept{
            return 7;
        }

        constexpr base_type const& operator[](std::size_t k) const{
            switch(k){
                case 0: return accuracy;
                case 1: return balanced_accuracy;
                case 2: return F1_score;
                case 3: return matthews_corr_coef;
                case 4: return fowlkes_mallows_idx;
                case 5: return informedness;
                case 6: return markedness;
            }
            throw std::runtime_error(
                "amt::classification::metrics::operator[](std::size_t) : "
                "invalid index. 0 <= index <= 6"
            );
        }

        constexpr base_type& operator[](std::size_t k){
            switch(k){
                case 0: return accuracy;
                case 1: return balanced_accuracy;
                case 2: return F1_score;
                case 3: return matthews_corr_coef;
                case 4: return fowlkes_mallows_idx;
                case 5: return informedness;
                case 6: return markedness;
            }
            throw std::runtime_error(
                "amt::classification::metrics::operator[](std::size_t) : "
                "invalid index. 0 <= index <= 6"
            );
        }

        constexpr std::string_view name(std::size_t k) const noexcept{
            switch(k){
                case 0: return "Accuracy";
                case 1: return "Balanced Accuracy";
                case 2: return "F1 Score";
                case 3: return "Matthews correlation coefficient";
                case 4: return "Fowlkes-Mallows index";
                case 5: return "Informedness";
                case 6: return "Markedness";
            }
            return "Unknown";
        }
    };

    double weighted_sum( SeriesViewOrSeries auto const& sample_score, bool normalize = false ){
        if(normalize)
            return mean(sample_score);
        else
            return sum(sample_score, std::plus<double>{});
    }
    
    double weighted_sum( SeriesViewOrSeries auto const& sample_score, SeriesViewOrSeries auto const& sample_weight, bool normalize = false ){
        double res{};
        std::size_t sz = sample_score.size();
        #pragma omp parallel for schedule(static) reduction( + : res )
        for(auto i = 0u; i < sz; ++i){
            double l = sample_score[i];
            double r = sample_weight[i];
            res += l * r;
        }
        if(normalize)
            return res / static_cast<double>(sz);
        else
            return res;
    }

    std::size_t count_nonzero(FrameViewOrFrame auto const& diff, std::size_t idx = 0u){
        auto const& s = diff[idx];
        std::size_t ct{};
        auto sz = s.size();
        #pragma omp parallel for schedule(static) reduction( + : res )
        for(auto i = 0u; i < sz; ++i){
            auto el = s[i].template as<double>();
            ct += (el != 0);
        }
        return ct;
    }

    double accuracy(FrameViewOrFrame auto const& y_pred, FrameViewOrFrame auto const& y_true, bool normalize = true){
        return accuracy(y_pred[0], y_true[0], normalize);
    }

    double accuracy(SeriesViewOrSeries auto y_pred, SeriesViewOrSeries auto const& y_true, bool normalize = true){
        if(y_pred.size() != y_true.size()){
            throw std::runtime_error("y_pred and y_true size mismatch");
        }
        auto size = y_pred.size();
        #pragma omp parallel for schedule(static)
        for(auto j = 0u; j < size; ++j){
            y_pred[j] = static_cast<double>( y_true[j] == y_pred[j] );
        }
        return weighted_sum(y_pred,normalize);
    }

    frame<> confusion_matrix(FrameViewOrFrame auto const& y_pred, FrameViewOrFrame auto const& y_true, std::size_t labels){
        return confusion_matrix(y_pred[0], y_true[0], labels);
    }

    frame<> confusion_matrix(SeriesViewOrSeries auto y_pred, SeriesViewOrSeries auto const& y_true, std::size_t labels){
        frame<> ret(labels,labels, double(0));
        for(auto i = 0u; i < y_pred.size(); ++i){
            double p = y_pred[i];
            double t = y_true[i];
            auto& el = ret[static_cast<std::size_t>(t)][static_cast<std::size_t>(p)];
            double prev = el;
            el = prev + 1.;
        }
        return ret;
    }

    std::string confusion_matrix_to_string(FrameViewOrFrame auto const& mat, std::vector<std::string> labels = {}, std::size_t indent = 3){
        std::stringstream ss;
        std::size_t max_width{};
        std::size_t number_max_width{};

        if( labels.size() != mat.cols() ){
            labels.resize(mat.cols());
            for(auto i = 0u; i < mat.cols(); ++i){
                labels[i] = std::to_string(i);
            }
        }

        for(auto const& el : labels){
            max_width = std::max(max_width, el.size());
        }

        auto temp = to<std::string>(mat,out_place);

        for(auto const& s : temp){
            for(auto const& el : s){
                std::string const& str = el;
                number_max_width = std::max(number_max_width, str.size());
            }
        }

        max_width += 2;
        
        ss<< std::string(max_width + indent, ' ');

        for(auto const& el : labels){
            auto w = std::max(number_max_width, el.size()) - el.size();
            ss << std::string(w, ' ') << std::quoted(el) << std::string(indent, ' ');
        }
        ss << '\n';
        auto sz = labels.size();
        for( auto i = 0u; i < sz; ++i ){
            auto const& el = labels[i];
            auto w = max_width - el.size();
            ss << std::quoted(el) << std::string(w + indent - 2, ' ');
            for(auto j = 0u; j < sz; ++j){
                std::string const& str = temp[j][i];
                auto diff = std::max(labels[j].size(), number_max_width) - str.size() + 1;
                auto start_sp = std::string(diff, ' ');
                auto end_sp = std::string(indent + 1, ' ');
                ss << start_sp << temp[j][i] << end_sp;
            }
            ss << '\n';
        }
        return ss.str();
    }

    void print_confusion_matrix(FrameViewOrFrame auto const& y_pred, FrameViewOrFrame auto const& y_true, std::vector<std::string> const& labels, std::size_t indent = 3){
        print_confusion_matrix(y_pred[0], y_true[0], labels, indent);
    }

    void print_confusion_matrix(SeriesViewOrSeries auto const& y_pred, SeriesViewOrSeries auto const& y_true, std::vector<std::string> const& labels, std::size_t indent = 3){
        auto conf = confusion_matrix(y_pred, y_true, labels.size());
        std::cout << confusion_matrix_to_string(conf, labels, indent);
    }

    namespace detail{

        std::vector<double> true_pos(FrameViewOrFrame auto const& mat){
            
            std::vector<double> res(mat.cols(),0);
            for(auto i = 0u; i < mat.cols(); ++i){
                res[i] = mat[i][i];
            }
            return res;
        }
        
        std::vector<double> true_neg(FrameViewOrFrame auto const& mat){
            auto cols = mat.cols();
            auto rows = mat.rows();
            std::vector<double> res(cols,0);
            for(auto i = 0u; i < cols; ++i){
                for(auto j = 0u; j < cols; ++j){
                    for(auto k = 0u; k < rows; ++k){
                        if(i == j or k == i) continue;
                        res[i] += static_cast<double>(mat[j][k]);
                    }
                }
            }
            return res;
        }

        std::vector<double> false_neg(FrameViewOrFrame auto const& mat){
            auto cols = mat.cols();
            auto rows = mat.rows();
            std::vector<double> res(cols,0);
            for(auto i = 0u; i < cols; ++i){
                for(auto j = 0u; j < rows; ++j){
                    if(i == j) continue;
                    res[i] += static_cast<double>(mat[i][j]);
                }
            }
            return res;
        }
        
        std::vector<double> false_pos(FrameViewOrFrame auto const& mat){
            auto cols = mat.cols();
            auto rows = mat.rows();
            std::vector<double> res(cols,0);
            for(auto i = 0u; i < rows; ++i){
                for(auto j = 0u; j < cols; ++j){
                    if(i == j) continue;
                    res[i] += static_cast<double>(mat[j][i]);
                }
            }
            return res;
        }
        
        std::vector<double> get_rate(std::vector<double> const& num, std::vector<double> const& den){
            auto res = num;
            std::transform(num.begin(), num.end(), den.begin(), res.begin(), std::divides<>{});
            return res;
        }
        
        
        // precision or positive predictive value (PPV)
        // PPV = TP / (TP + FP)
        std::vector<double> TNR(std::vector<double> const& tn, std::vector<double> const& n){
            return get_rate(tn,n);
        }

    } // namespace detail

    metrics metrics_from_confusion_matrix(FrameViewOrFrame auto const& mat){
        
        metrics res;

        auto TP = detail::true_pos(mat);
        auto TN = detail::true_neg(mat);
        auto FP = detail::false_pos(mat);
        auto FN = detail::false_neg(mat);
        auto P = TP;
        auto N = TN;
        auto sz = mat.cols();


        std::transform(TP.begin(), TP.end(), FN.begin(), P.begin(), std::plus<>{});
        std::transform(TN.begin(), TN.end(), FP.begin(), N.begin(), std::plus<>{});

        // sensitivity, recall, hit rate, or true positive rate (TPR)
        // TPR = TP / (TP + FN); condition positive (P) = TP + FN
        auto TPR = detail::get_rate(TP,P);

        // specificity, selectivity or true negative rate (TNR)
        // TNR = TN / (TN + FP); condition positive (N) = TN + FP
        auto TNR = detail::get_rate(TN, N);

        // negative predictive value (NPV)
        // NPV = TN / (TN + FN)
        auto NPV = TN;
        for(auto i = 0u; i < sz; ++i){
            NPV[i] = TN[i] / ( TN[i] + FN[i] );
        }

        // precision or positive predictive value (PPV)
        // PPV = TP / (TP + FP)
        auto PPV = TP;
        for(auto i = 0u; i < sz; ++i){
            PPV[i] = TP[i] / ( TP[i] + FP[i] );
        }

        auto one_minus_value = [](auto const& val){
            return 1 - val;
        };

        // miss rate or false negative rate (FNR)
        // FNR = 1 - TPR
        auto FNR = TPR;
        std::transform(TPR.begin(), TPR.end(), FNR.begin(), one_minus_value);

        // fall-out or false positive rate (FPR)
        // FPR = 1 - TNR
        auto FPR = TNR;
        std::transform(TNR.begin(), TNR.end(), FPR.begin(), one_minus_value);

        // false discovery rate (FDR)
        // FDR = 1 - PPV
        auto FDR = PPV;
        std::transform(PPV.begin(), PPV.end(), FDR.begin(), one_minus_value);

        // false omission rate (FOR)
        // FOR = 1 - NPV
        auto FOR = NPV;
        std::transform(NPV.begin(), NPV.end(), FOR.begin(), one_minus_value);

        // Prevalence Threshold (PT)
        auto PT = TPR;
        for(auto i = 0u; i < sz; ++i){
            auto root = std::sqrt( TPR[i] * (-TNR[i] + 1.) );
            auto num = root + TNR[i] - 1.;
            PT[i] = num / (TPR[i] + TNR[i] - 1);
        }

        // Threat score (TS) or critical success index (CSI)
        // TS = TP / (TP + FN + FP)
        auto TS = TP;
        for(auto i = 0u; i < sz; ++i){
            TS[i] = TP[i] / ( TP[i] + FN[i] + FP[i] );
        }

        // accuracy (ACC)
        res.accuracy = TP;
        for(auto i = 0u; i < sz; ++i){
            res.accuracy[i] = (TP[i] + TN[i]) / (P[i] + N[i]);
        }

        // balanced accuracy (BA)
        res.balanced_accuracy = TPR;
        for(auto i = 0u; i < sz; ++i){
            res.balanced_accuracy[i] = ( TPR[i] + TNR[i] ) / 2.;
        }

        // F1 score
        res.F1_score = TPR;
        for(auto i = 0u; i < sz; ++i){
            res.F1_score[i] = 2. * ( ( PPV[i] * TPR[i] ) / ( PPV[i] + TPR[i] ) );
        }

        // Matthews correlation coefficient (MCC)
        res.matthews_corr_coef = TPR;
        for(auto i = 0u; i < sz; ++i){
            auto den = (TP[i] + FP[i]) * (TP[i] + FN[i]) * (TN[i]+ FP[i]) * (TN[i] + FN[i]);
            res.matthews_corr_coef[i] = ( TP[i] * TN[i] - FP[i] * FN[i]  ) / std::sqrt( den );
        }
        
        // Fowlkes–Mallows index (FM)
        res.fowlkes_mallows_idx = PPV;
        for(auto i = 0u; i < sz; ++i){
            res.fowlkes_mallows_idx[i] = std::sqrt( PPV[i] * TPR[i] );
        }
        
        // informedness or bookmaker informedness (BM)
        res.informedness = TPR;
        for(auto i = 0u; i < sz; ++i){
            res.informedness[i] = TPR[i] + TNR[i] - 1;
        }
        
        // markedness (MK) or deltaP
        res.markedness = PPV;
        for(auto i = 0u; i < sz; ++i){
            res.markedness[i] = PPV[i] + NPV[i] - 1;
        }

        return res;
    }

    metrics calculate_metrics(SeriesViewOrSeries auto y_pred, SeriesViewOrSeries auto const& y_true, std::size_t labels){
        auto conf = confusion_matrix(y_pred, y_true, labels);
        return metrics_from_confusion_matrix(conf);
    }

    metrics calculate_metrics(FrameViewOrFrame auto const& y_pred, FrameViewOrFrame auto const& y_true, std::size_t labels){
        auto conf = confusion_matrix(y_pred[0], y_true[0], labels);
        return metrics_from_confusion_matrix(conf);
    }

    std::string metrics_to_string(metrics const& m, std::vector<std::string> labels = {}, std::size_t indent = 3){
        std::stringstream ss;
        std::size_t max_width{};
        std::size_t number_max_width{};

        if( labels.size() != m[0].size() ){
            labels.resize(m[0].size());
            for(auto i = 0u; i < m[0].size(); ++i){
                labels[i] = std::to_string(i);
            }
        }
        
        for(auto i = 0u; i < m.total(); ++i){
            max_width = std::max(max_width, m.name(i).size());

            for(auto const& el : m[i])
                number_max_width = std::max(number_max_width, std::to_string(el).size());
        }

        ss<< std::string(max_width + indent + 2, ' ');

        for(auto const& el : labels){
            auto w = std::max(number_max_width, el.size()) - el.size();
            ss << std::string(w, ' ') << std::quoted(el) << std::string(indent, ' ');
        }

        ss << '\n';

        for(auto i = 0u; i < m.total(); ++i){
            auto name = m.name(i);
            auto w = max_width - name.size() + indent;
            ss << std::quoted(name) << std::string(w, ' ');
            auto const& v = m[i];
            for(auto j = 0u; j < v.size(); ++j){
                auto const& el = std::to_string(v[j]);
                auto lsz = std::max(number_max_width, labels[j].size());
                auto start =  std::string(lsz - el.size() + 1u, ' ');
                auto end =  std::string(indent + 1u, ' ');
                ss<<start<<el<<end;
            }
            ss << '\n';
        }
        
        return ss.str();
    }

    std::ostream& operator<<(std::ostream& os, metrics const& m){
        return os << metrics_to_string(m,m.labels);
    }

    void print_metrics(SeriesViewOrSeries auto y_pred, SeriesViewOrSeries auto const& y_true, std::vector<std::string> labels){
        auto conf = confusion_matrix(y_pred, y_true, labels.size());
        auto m = metrics_from_confusion_matrix(conf);
        std::cout<<confusion_matrix_to_string(conf,labels)<<'\n';
        std::cout<<metrics_to_string(m,std::move(labels));
    }

    void print_metrics(FrameViewOrFrame auto const& y_pred, FrameViewOrFrame auto const& y_true, std::vector<std::string> labels){
        print_metrics(y_pred[0], y_true[0], std::move(labels));
    }

} // namespace amt::classification

#endif // AMT_METRICS_CLASSIFICATION_HPP

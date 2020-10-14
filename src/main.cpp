 #include <algorithm>
#include <frame.hpp>
#include <functional>
#include <functions.hpp>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <dataframe.hpp>
#include <matplotlibcpp.h>
#include <model/LogisticRegression/logistic_regression.hpp>

// void clean_data(amt::frame<>& f){
//     amt::name_list n = {
//         "id",
//         "date",
//         // "price",
//         "bedrooms",
//         "bathrooms",
//         // "sqft_living",
//         "sqft_lot",
//         "floors",
//         "waterfront",
//         "view",
//         "condition",
//         "grade",
//         "sqft_above",
//         "sqft_basement",
//         "yr_built",
//         "yr_renovated",
//         "zipcode",
//         "lat",
//         "long",
//         "sqft_living15",
//         "sqft_lot15" 
//     };
//     amt::drop_cols(f,amt::in_place, std::move(n));
//     amt::filter(f,amt::in_place,[](std::string_view val){
//         std::string temp(val);
//         std::transform(temp.begin(),temp.end(), temp.begin(), [](auto c){return std::tolower(c);});
//         if( temp.empty() ||  (temp == "nan") ) return true;
//         return false;
//     });

//     amt::to<double>(f,amt::in_place);
// }

inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}


inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}


inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}
std::string norm_str(std::string name){
    trim(name);
    name[0] = static_cast<char>(std::toupper(name[0]));
    std::transform(name.begin(),name.end(),name.begin(),[](auto const& c){
        if( c == '_' ) return ' ';
        return c;
    });
    for(auto i = 2ul; i < name.size(); ++i){
        auto& p = name[i - 1];
        auto& c = name[i];
        if(p == ' ') c = static_cast<char>(std::toupper(c));
    }
    return name;
}

// void plot(amt::frame<> const& x, amt::frame<> const& y){
//     namespace plt = matplotlibcpp;
//     std::vector<double> temp_x(x.rows()), temp_y(y.rows());

//     auto i = 0ul;
//     for(auto const& el : y[0]){
//         auto val = el.template as<double>();
//         temp_y[i++] = val;
//     }

//     i = 0ul;
//     for(auto const& el : x[0]){
//         auto val = el.template as<double>();
//         temp_x[i++] = val;
//     }
//     // plt::figure_size(1200, 780);
//     plt::title("House Pricing");
//     plt::scatter(temp_x,temp_y);
//     plt::xlabel( norm_str(x.name(0)) );
//     plt::ylabel( norm_str(y.name(0)) );
//     plt::show();
// }

// void plot_pred(double s, double c, amt::frame<> const& x, amt::frame<> const& y){
//     namespace plt = matplotlibcpp;
//     std::vector<double> temp_x(x.rows()), temp_y(y.rows());

//     auto i = 0ul;
//     auto max = 0.0;
//     for(auto const& el : y[0]){
//         auto val = el.template as<double>();
//         temp_y[i++] = val;
//     }

//     i = 0ul;
//     for(auto const& el : x[0]){
//         auto val = el.template as<double>();
//         temp_x[i++] = val;
//         max = std::max(val,max);
//     }

//     auto sz = x.rows();
//     auto chunks = max / static_cast<double>(x.rows());
//     std::vector<double> range_x(sz), temp_p(sz);
    
//     for(auto k = 1u; k < sz; ++k) range_x[k] = range_x[k - 1u] + chunks;

//     for(auto j = 0ul; j < sz; ++j) 
//         temp_p[j] = c + s * static_cast<double>(range_x[j]);
//     // plt::figure_size(1200, 780);
//     plt::title("House Pricing");
//     plt::scatter(temp_x,temp_y);
//     plt::plot(range_x,temp_p,"r");
//     plt::xlabel( norm_str(x.name(0)) );
//     plt::ylabel( norm_str(y.name(0)) );
//     plt::show();
// }

// int main(int, char **) {

//     auto temp = amt::read_csv("/Users/amit/Desktop/code/dataframe/src/test.csv", true);
//     amt::to<float>(temp,amt::in_place);
//     clean_data(temp);
    
//     auto i = 0ul;
//     auto y = amt::drop_cols(temp,amt::out_place,[&i](auto const&){
//         if(i++ > 0) return true;
//         return false;
//     });
//     auto x = amt::drop_cols(temp,amt::out_place,amt::index_list{0});

//     auto split_ratio = 0.25;
//     auto rsz = y.rows();
//     auto training_sz = static_cast<std::size_t>(split_ratio * static_cast<double>(rsz));
//     // auto testing_sz = rsz - training_sz;

//     for(auto& s : x){
//         double ma = amt::max(s);
//         double mi = amt::min(s);
//         double de = ma - mi;
//         amt::transform(s,amt::in_place,[&de,&mi]<typename T>(T& val){
//             if constexpr( std::is_convertible_v<T,double> )
//                 return (val - static_cast<T>(mi)) / static_cast<T>(de);
//             else
//                 return val;
//         });
//     }
//     for(auto& s : y){
//         auto ma = amt::max(s);
//         auto mi = amt::min(s);
//         auto de = ma - mi;
//         amt::transform(s,amt::in_place,[&de,&mi]<typename T>(T& val){
//             if constexpr( std::is_convertible_v<T,double> )
//                 return (val - static_cast<T>(mi)) / static_cast<T>(de);
//             else
//                 return val;
//         });
//     }

//     auto x_train = amt::drop_rows(x,amt::out_place,training_sz);
//     auto y_train = amt::drop_rows(y,amt::out_place,training_sz);

//     auto x_test = amt::drop_rows(x,amt::out_place,0, training_sz);
//     auto y_test = amt::drop_rows(y,amt::out_place,0, training_sz);
//     // plot(x,y);

//     auto l_model = amt::LinearRegression(x_train,y_train, 0, amt::linear_regression::gradient_descent{0.5,1500});
//     plot_pred(l_model.beta(1), l_model.beta(0), x_test, y_test);
//     // auto pred = l_model.predict(x_test);
//     // y_test.push_back(std::move(pred));
//     std::cout<<l_model.beta()<<'\n';

//     return 0;
// }

auto map_col(amt::Series auto& s){
    std::unordered_map<std::string,double> ret;
    double i = 0.0;
    for(auto& el : s){
        amt::visit(el, [&ret,&i,&el](std::string& s){
            if(auto it = ret.find(s); it != ret.end()){
                el = it->second;
            }else{
                ret.insert({s, i});
                el = i++;
            }
        });
    }
    return ret;
}

auto preprocess(amt::Frame auto& f){

    amt::index_list ids;
    auto& s = f["Species"];
    for(auto i = 0u; i < f.rows(); ++i){
        auto& el = s[i];
        if(el.template as<std::string>() == "Iris-virginica") ids.insert(i);
    }
    amt::drop_rows(f,amt::in_place,std::move(ids));
    auto n = amt::name_list{
        "Id",
        // "SepalLengthCm"
        // "SepalWidthCm"
        "PetalLengthCm",
        "PetalWidthCm",
        "Species"
    };
    auto x = amt::drop_cols(f,amt::out_place, std::move(n));
    amt::to<double>(x,amt::in_place);
    auto y = amt::frame<>({f["Species"]});
    return std::make_pair(x,y);
}

void plot(amt::series<> const& x, amt::series<> const& y, amt::series<> const& target){
    namespace plt = matplotlibcpp;
    plt::title("Logistic Reg");
    plt::xlabel( norm_str(x.name()) );
    plt::ylabel( norm_str(y.name()) );

    for(auto i = 0ul; i < x.size(); ++i){
        auto xel = x[i].template as<double>();
        auto yel = y[i].template as<double>();
        auto tel = target[i].template as<double>();
        plt::plot({xel},{yel}, { 
            {"marker", "o"}, 
            {"linestyle",""}, 
            {"color",tel == 0.0 ? "r" : "b"}
        });
    }

    // plt::figure_size(1200, 780);
    plt::show();
}

void plot_pred(amt::LogisticRegression const&, amt::series<> const& x, amt::series<> const& y, amt::series<> const& target){
    namespace plt = matplotlibcpp;
    plt::title("Logistic Reg");
    plt::xlabel( norm_str(x.name()) );
    plt::ylabel( norm_str(y.name()) );
    auto sz = x.size();

    // auto x_max = amt::max(x);
    // auto x_min = amt::min(x);
    
    // auto y_max = amt::max(y);
    // auto y_min = amt::min(y);

    // auto lsz = 10;

    // arma::Col<double> ones = arma::ones(lsz);
    // arma::Col<double> xl = arma::linspace(x_min,x_max,lsz);
    // arma::Col<double> yl = arma::linspace(y_min,y_max,lsz);

    // arma::Mat<double> x_range = arma::join_rows(ones, arma::join_rows(xl,yl));
    // arma::Mat<double> y_prob = model.predict(x_range);

    // std::vector<std::vector<double>> xx(sz,std::vector<double>(sz));
    // std::vector<std::vector<double>> yy(sz,std::vector<double>(sz));
    // std::vector<std::vector<double>> y_pred(sz,std::vector<double>(1));

    // for(auto i = 0u; i < lsz; ++i){
    //     xl = arma::join_cols(xl,)
    // }

    // for(auto i = 0u; i < sz; ++i){
    //     for(auto j = 0u; j < sz; ++j){
    //         xx[i][j] = xl[j];
    //         xx[j][i] = yl[i];
    //     }
    // }

    // for(auto i = 0u; i < sz; ++i) y_pred[i][0] = y_prob(i,0);

    for(auto i = 0ul; i < sz; ++i){
        auto xel = x[i].template as<double>();
        auto yel = y[i].template as<double>();
        auto tel = target[i].template as<double>();
        plt::plot({xel},{yel}, { 
            {"marker", "o"}, 
            {"linestyle",""}, 
            {"color",tel == 0.0 ? "r" : "b"},
        });
    }

    // plt::contour(xx,yy,y_pred);
    // plt::figure_size(1200, 780);
    plt::show();
}

double cal_err(amt::FrameViewOrFrame auto const& y_p, amt::FrameViewOrFrame auto const& y){
    using frame_type = std::decay_t<decltype(y_p)>;
    frame_type diff = (y_p - y);
    double m = static_cast<double>(y.rows());
    double sum = amt::accumulate(diff, 0.0, [](double r, double v){
        return r + ( v < 0 ? 0 : 1 );
    });
    return ( sum / m ) * 100.0;
}

double cal_err(amt::FrameViewOrFrame auto const& y){
    auto diff = *(y[0] - y[1]);
    double m = static_cast<double>(y.rows());
    double sum = amt::accumulate(diff, 0.0, [](double r, double v){
        return r + ( v < 0 ? 0 : 1 );
    });
    return ( sum / m ) * 100.0;
}

int main(){
    auto filename = "/Users/amit/Desktop/code/ML/ML-v2/dataset/Iris.csv";
    auto temp = amt::read_csv(filename, true);

    amt::shuffle(temp);

    auto [X,Y] = preprocess(temp);
    auto target_name = map_col(Y[0]);

    auto ratio = 0.75;
    auto training_sz = static_cast<std::size_t>(static_cast<double>(X.rows()) * ratio);

    auto x_train = amt::drop_rows(X,amt::out_place,training_sz);
    auto y_train = amt::drop_rows(Y,amt::out_place,training_sz);

    auto x_test = amt::drop_rows(X,amt::out_place,0, training_sz);
    auto y_test = amt::drop_rows(Y,amt::out_place,0, training_sz);
    
    auto model = amt::LogisticRegression(x_train,y_train,amt::logistic_regression::gradient_descent{0.01,1500});
    // amt::to<double>(Y,amt::in_place);
    // std::cout<<model.beta()<<'\n';
    // plot_pred(model,X[0],X[1],Y[0]);
    auto y_pred = model.predict(x_test);
    y_pred.push_back(y_test);
    std::cout<<y_pred<<'\n';
    std::cout<<cal_err(y_pred)<<'\n';
}

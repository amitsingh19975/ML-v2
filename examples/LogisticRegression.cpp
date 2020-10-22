 #include <algorithm>
#include <functional>
#include <functions.hpp>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <dataframe.hpp>
#include <matplotlibcpp.h>
#include <model/LogisticRegression/logistic_regression.hpp>
#include <metrics/classification.hpp>

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

auto map_to_vec(std::unordered_map<std::string,double> const& m){
    std::vector<std::string> ret(m.size());
    for(auto const& [key,val] : m){
        std::size_t i = static_cast<std::size_t>(val);
        ret[i] = key;
    }
    return ret;
}

auto preprocess(amt::Frame auto& f){

    // amt::index_list ids;
    // auto& s = f["Species"];
    // for(auto i = 0u; i < f.rows(); ++i){
    //     auto& el = s[i];
    //     if(el.template as<std::string>() == "Iris-virginica") ids.insert(i);
    // }
    // amt::drop_rows(f,amt::in_place,std::move(ids));
    auto n = amt::name_list{
        "Id",
        // "SepalLengthCm"
        // "SepalWidthCm"
        // "PetalLengthCm",
        // "PetalWidthCm",
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

template<typename T>
void plot_pred(amt::LogisticRegression<T> const&, amt::series<> const& x, amt::series<> const& y, amt::series<> const& target){
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
        return r + ( v == 0 );
    });
    return ( sum / m ) * 100.0;
}

double cal_err(amt::FrameViewOrFrame auto const& y){
    auto diff = *(y[0] - y[1]);
    double m = static_cast<double>(y.rows());
    double sum = amt::accumulate(diff, 0.0, [](double r, double v){
        return r + ( v == 0 );
    });
    return ( sum / m ) * 100.0;
}

int main(){
    auto filename = "/Users/amit/Desktop/code/ML/ML-v2/dataset/Iris.csv";
    auto temp = amt::read_csv(filename, true);

    amt::shuffle(temp,static_cast<std::size_t>(std::time(0)));
    // amt::shuffle(temp);

    auto [X,Y] = preprocess(temp);
    auto target_name = map_col(Y[0]);

    auto ratio = 0.25;
    auto training_sz = static_cast<std::size_t>(static_cast<double>(X.rows()) * ratio);

    auto x_train = amt::drop_rows(X,amt::out_place,training_sz);
    auto y_train = amt::drop_rows(Y,amt::out_place,training_sz);

    auto x_test = amt::drop_rows(X,amt::out_place,0, training_sz);
    auto y_test = amt::drop_rows(Y,amt::out_place,0, training_sz);
    
    auto model = amt::LogisticRegression<amt::OVR>(x_train,y_train);
    // amt::to<double>(Y,amt::in_place);
    // std::cout<<model.beta()<<'\n';
    // plot_pred(model,X[0],X[1],Y[0]);
    
    auto y_pred = model.predict(x_test);
    // std::cout<<model.beta()<<'\n';
    // y_pred.push_back(y_test);
    // std::cout<<y_pred<<'\n';
    auto labels = map_to_vec(target_name);
    amt::classification::print_metrics(y_pred,y_test,labels);
}

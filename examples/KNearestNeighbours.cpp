 #include <algorithm>
#include <functional>
#include <functions.hpp>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <dataframe.hpp>
#include <model/KAlgorithms/k_nearest_neighbours.hpp>
#include <metrics/classification.hpp>
#include <matplot/matplot.h>

namespace plt = matplot;

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

auto map_col(amt::PureSeries auto& s){
    std::unordered_map<std::string,double> ret;
    double i = 0.0;
    s.reset_dtype();
    for(auto& el : s){
        auto &str = get<std::string>(el);
        if(auto it = ret.find(str); it != ret.end()){
            el = it->second;
        }else{
            ret.insert({str, i});
            el = i++;
        }
    }
    s.dtype(amt::dtype<double>());
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
    // amt::drop_row(f,amt::tag::inplace,std::move(ids));
    auto n = amt::name_list{
        "Id",
        // "SepalLengthCm"
        // "SepalWidthCm"
        // "PetalLengthCm",
        // "PetalWidthCm",
        "Species"
    };
    auto x = amt::drop_col(f, std::move(n));
    amt::cast<double>(x,amt::tag::inplace);
    auto y = amt::frame({f["Species"]});
    return std::make_pair(x,y);
}


void plot(amt::frame const& x, amt::series const& tar){
    
    auto xv = amt::to_vector<double>(x[0]);
    auto yv = amt::to_vector<double>(x[1]);
    auto zv = amt::to_vector<double>(x[2]);
    auto c = amt::to_vector<double>(tar);
    std::vector<double> sizes(xv.size(), 8);

    auto l = plt::scatter3(xv, yv, zv, sizes, c);
    l->marker_face(true);
    l->marker_style(plt::line_spec::marker_style::diamond);

    plt::show();
}

int main(){
    auto filename = "/Users/amit/Desktop/code/ML/ML-v2/dataset/Iris.csv";
    auto temp = amt::read_csv(filename, true);

    amt::shuffle(temp, amt::tag::inplace, static_cast<unsigned>(std::time(0)));
    // amt::shuffle(temp);

    auto [X,Y] = preprocess(temp);
    auto target_name = map_col(Y[0]);

    auto ratio = 0.25;
    auto training_sz = static_cast<std::size_t>(static_cast<double>(X.rows()) * ratio);

    auto x_train = amt::drop_row(X,training_sz);
    auto y_train = amt::drop_row(Y,training_sz);

    auto x_test = amt::drop_row(X,0, training_sz);
    auto y_test = amt::drop_row(Y,0, training_sz);
    
    auto model = amt::classification::KNearestNeighbours(x_train,y_train,4);
    auto y_pred = model.predict(x_test);
    // plot(X,Y[0]);

    auto labels = map_to_vec(target_name);
    amt::classification::print_metrics(y_pred,y_test,labels);
    
}

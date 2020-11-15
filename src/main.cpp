 #include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <dataframe.hpp>
#include <model/LogisticRegression/logistic_regression.hpp>
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
    //     if(amt::get<std::string>(el) == "Iris-virginica") ids.push_back(i);
    // }
    // amt::drop_row(f,amt::tag::inplace,std::move(ids));
    auto n = amt::name_list{
        "Id",
        // "SepalLengthCm"
        // "SepalWidthCm"
        "PetalLengthCm",
        "PetalWidthCm",
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

template<typename T>
void print_v(std::vector<T> const& v){
    for(auto const& el : v){
        std::cout<< el <<", ";
    }
}

#include <model/DecisionTree/decision_tree.hpp>

int main(){
    // auto filename = "/Users/amit/Desktop/code/ML/ML-v2/dataset/Iris.csv";
    // auto temp = amt::read_csv(filename, true);
    amt::frame temp = {
        { "Day", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, amt::dtype<std::int64_t>{} },
        { "Outlook", {"Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"}, amt::dtype<std::string>{} },
        { "Temperature", {"Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"}, amt::dtype<std::string>{} },
        { "Humidity", {"High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"}, amt::dtype<std::string>{} },
        { "Wind", {"Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"}, amt::dtype<std::string>{} },
        { "Plays", {"No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"}, amt::dtype<std::string>{} }
    };
    amt::shuffle(temp, amt::tag::inplace, static_cast<unsigned>(std::time(0)));
    
    amt::drop_col(temp, amt::tag::inplace, amt::index_list{0});

    std::cout<<amt::pretty_string(temp)<<'\n';

    std::vector< std::vector<amt::box> > mp(temp.cols());
    for(auto i = 0ul; i < temp.cols(); ++i){
        auto [cat, _] = amt::sorted_factorize<>(temp[i], amt::tag::inplace);
        mp[i] = std::move(cat);
    }

    // for(auto const& el : mp) {
    //     std::cout<<"[ ";
    //     print_v(el);
    //     std::cout<<"]\n";
    // }
    amt::classification::DecisionTree<amt::splitter::ID3> t(temp);

    amt::frame p = {
        {"Outlook", {1,1}, amt::dtype<double>()},
        {"Wind", {1,0}, amt::dtype<double>()}
    };
    auto prd = t.predict(p);
    p.col_push_back(std::move(prd[0]));
    
    for(auto i = 0ul; i < p.cols() - 1u; ++i){
        std::unordered_map<amt::box,amt::box> map;
        auto& s = p[i];
        auto idx = temp.name_index(s.name());
        for(auto j = 0u; j < mp[idx].size(); ++j){
            map[static_cast<double>(j)] = mp[idx][j];
        }
        amt::replace(s, amt::tag::inplace, std::move(map));
    }
    std::unordered_map<amt::box,amt::box> map;
    auto idx = temp.cols() - 1u;
    for(auto j = 0u; j < mp[idx].size(); ++j){
        map[static_cast<double>(j)] = mp[idx][j];
    }
    amt::replace(p.back(), amt::tag::inplace, std::move(map));
    std::cout<<amt::pretty_string(p)<<'\n';
    

}

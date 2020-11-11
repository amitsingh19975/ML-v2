 #include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <dataframe.hpp>
#include <model/LinearRegression/linear_regression.hpp>
#include <metrics/regression.hpp>

void clean_data(amt::frame& f){
    amt::name_list n = {
        "id",
        "date",
        // "price",
        "bedrooms",
        "bathrooms",
        // "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15" 
    };
    amt::drop_col(f,amt::tag::inplace, std::move(n));
    amt::cast<double>(f,amt::tag::inplace);
    amt::drop_nan(f, amt::tag::inplace);
}

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

// void plot(amt::frame const& x, amt::frame const& y){
//     namespace plt = matplotlibcpp;
//     std::vector<double> temp_x(x.rows()), temp_y(y.rows());

//     auto i = 0ul;
//     for(auto const& el : y[0]){
//         auto val = amt::get<double>(el);
//         temp_y[i++] = val;
//     }

//     i = 0ul;
//     for(auto const& el : x[0]){
//         auto val = amt::get<double>(el);
//         temp_x[i++] = val;
//     }
//     // plt::figure_size(1200, 780);
//     plt::title("House Pricing");
//     plt::scatter(temp_x,temp_y);
//     std::string x_str(x.name(0));
//     std::string y_str(y.name(0));
//     plt::xlabel( norm_str(x_str) );
//     plt::ylabel( norm_str(y_str) );
//     plt::show();
// }

// void plot_pred(double s, double c, amt::frame const& x, amt::frame const& y){
//     namespace plt = matplotlibcpp;
//     std::vector<double> temp_x(x.rows()), temp_y(y.rows());

//     auto i = 0ul;
//     auto max = 0.0;
//     for(auto const& el : y[0]){
//         auto val = amt::get<double>(el);
//         temp_y[i++] = val;
//     }

//     i = 0ul;
//     for(auto const& el : x[0]){
//         auto val = amt::get<double>(el);
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
//     std::string x_str(x.name(0));
//     std::string y_str(y.name(0));
//     plt::xlabel( norm_str(x_str) );
//     plt::ylabel( norm_str(y_str) );
//     plt::show();
// }

int main(int, char **) {

    auto temp = amt::read_csv("/Users/amit/Desktop/code/ML/ML-v2/dataset/house_pricing.csv", true);
    amt::infer<>(temp,amt::tag::inplace);
    clean_data(temp);
    
    auto y = amt::frame{temp[0]};
    auto x = amt::drop_col(temp,amt::index_list{0});

    auto split_ratio = 0.25;
    auto rsz = y.rows();
    auto training_sz = static_cast<std::size_t>(split_ratio * static_cast<double>(rsz));
    // auto testing_sz = rsz - training_sz;

    amt::minmax_norm<>(x, amt::tag::inplace);
    amt::minmax_norm<>(y, amt::tag::inplace);

    auto x_train = amt::drop_row(x,training_sz);
    auto y_train = amt::drop_row(y,training_sz);

    auto x_test = amt::drop_row(x,0, training_sz);
    auto y_test = amt::drop_row(y,0, training_sz);
    // plot(x,y);
    auto l_model = amt::regression::LinearRegression(x_train,y_train, 0, amt::regression::gradient_descent{0.5,1500});
    // plot_pred(l_model.beta(1), l_model.beta(0), x_test, y_test);
    auto pred = l_model.predict(x_test);
    // y_test.push_back(std::move(pred));
    // std::cout<<l_model.beta()<<'\n';
    std::cout<<amt::regression::calculate_metrics(pred,y_test);
    return 0;
}
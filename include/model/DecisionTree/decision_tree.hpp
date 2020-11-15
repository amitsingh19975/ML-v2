#if !defined(AMT_MODEL_DECISION_TREE_DECISION_TREE_HPP)
#define AMT_MODEL_DECISION_TREE_DECISION_TREE_HPP

#include <model/DecisionTree/splitter.hpp>
#include <dataframe.hpp>
#include <memory>

namespace amt::classification{

namespace impl{
    struct DecisionNode{
        union {
            double value;
            std::size_t idx;
        };
        std::string name;
        bool is_leaf{false};
        std::vector< std::unique_ptr<DecisionNode> > next{};
    };
} // namespace impl

template<typename Splitter>
struct DecisionTree{
    
    using node_type = impl::DecisionNode;

    DecisionTree() = default;
    DecisionTree(DecisionTree const &other) = default;
    DecisionTree(DecisionTree &&other) = default;
    DecisionTree &operator=(DecisionTree const &other) = default;
    DecisionTree &operator=(DecisionTree &&other) = default;
    ~DecisionTree() = default;

    DecisionTree( Frame auto X, Frame auto const& Y)
        : m_splitter(Y)
    {
        concat_col(X,Y, amt::tag::inplace);
        fit(X, &m_node);
    }

    DecisionTree( Frame auto const& XY )
        : m_splitter(XY.back())
    {
        fit(XY, &m_node);
    }

    template<Frame FrameType>
    auto predict(FrameType const& X) const {
        frame_result_t<FrameType> ret(1ul, X.rows());
        ret.dtype(dtype<double>());
        ret[0].name("Predicted");
        for(auto i = 0ul; i < X.rows(); ++i){
            ret[0][i] = double(NAN);
            predict_helper(ret[0][i], X, &m_node, i);
        }
        return ret;
    }

private:

    void predict_helper(Box auto& res, Frame auto const& f, node_type const* root, std::size_t i) const{
        if(root == nullptr) return;

        if(root->is_leaf){
            res = root->value;
            return;
        }else{
            if(f.has_name(root->name)){
                auto const& s = f[root->name];
                double el = s[i];
                auto id = static_cast<std::size_t>(el);
                auto& ref = root->next[id];
                predict_helper(res, f, ref.get(), i);
            }
        }
    }

    void fit(Frame auto const& XY, node_type* root){
        if(XY.empty() || root == nullptr ) return;

        auto yuq = unique(XY.back());
        if(yuq.size() == 1u){
            root->value = *(yuq.begin());
            root->is_leaf = true;
            return;
        }
        
        if( XY.cols() - m_blocked_ids.size() == 1ul ){
            root->value = *std::max_element(yuq.begin(), yuq.end());
            root->is_leaf = true;
            return;
        }

        auto idx = m_splitter(XY, m_blocked_ids);
        auto uq = unique(XY[idx]);

        auto N = static_cast<std::size_t>(static_cast<double>(*std::max_element(uq.begin(), uq.end())));

        root->idx = idx;
        root->name = XY[idx].name();
        root->next.resize(N + 1u);
        m_blocked_ids.insert(root->idx);
        
        for(auto const& el : uq){
            auto tempf = filter[idx](XY, equal(el));
            auto tidx = static_cast<std::size_t>(static_cast<double>(el));
            root->next[tidx] = std::make_unique<node_type>();
            fit(tempf, root->next[tidx].get());
        }
    }

private:
    node_type m_node;
    Splitter m_splitter;
    std::unordered_set<std::size_t> m_blocked_ids;
};

} // namespace amt::classification


#endif // AMT_MODEL_DECISION_TREE_DECISION_TREE_HPP

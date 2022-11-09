#ifndef SCALERCOMPUTATIONGRAPH_HPP
#define SCALERCOMPUTATIONGRAPH_HPP

#include "ComputationGraph.hpp"

namespace DataStructures
{
    template <class T>
    class ScalerComputationGraph : public ComputationGraph<T>
    {
    private:
        class ScalerComputationGraphNode : public ComputationGraph<T>::ComputationGraphNode
        {
        public:
            ScalerComputationGraphNode(const std::string& nodeName = "ScalerComputationGraphNode");

            void Reset() override;

            void UpdateGradient(const T& partialGradient) override;
        };

        class ConstantNode : public ScalerComputationGraphNode
        {
        protected:
            T value;

        public:
            ConstantNode(const T& value, const std::string& nodeName = "ConstantNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class VariableNode : public ConstantNode
        {
        public:
            VariableNode(const T& value, const std::string& nodeName = "VariableNode");

            void SetValue(const T& newValue);
        };

        class FunctionNode : public ScalerComputationGraphNode
        {
        protected:
            class ComputationGraph<T>::ComputationGraphNode *firstInput;
            class ComputationGraph<T>::ComputationGraphNode *secondInput;

        public:
            FunctionNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName = "FunctionNode");

            typename ComputationGraph<T>::ComputationGraphNode *GetFirstInput() const;

            typename ComputationGraph<T>::ComputationGraphNode *GetSecondInput() const;
        };

        class AddNode : public FunctionNode
        {
        public:
            AddNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "AddNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MinusNode : public FunctionNode
        {
        public:
            MinusNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "MinusNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MultiplyNode : public FunctionNode
        {
        public:
            MultiplyNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "ScalerMultiplyNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class DivideNode : public FunctionNode
        {
        public:
            DivideNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "DivideNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class PowerNode : public FunctionNode
        {
        public:
            PowerNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "PowerNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        List<FunctionNode *> funcNodes;

    public:
        ScalerComputationGraph();

        T Forward() override;

        void Backward() override;

        void SetValue(const ComputationGraphNodeHandler<T>& handler, const T& newValue) const override;

        ComputationGraphNodeHandler<T> CreateConstantNode(const T& value, const std::string& name) override;

        ComputationGraphNodeHandler<T> CreateVariableNode(const T& value, const std::string& name) override;

        ComputationGraphNodeHandler<T> CreateFunctionNode(
            const ComputationGraphNodeHandler<T>& inputNodeHandler1,
            const ComputationGraphNodeHandler<T>& inputNodeHandler2,
            const GraphOperation& operation,
            const std::string& name) override;
    };
} // namespace DataStructure

namespace DataStructures
{
    template <class T>
    ScalerComputationGraph<T>::ScalerComputationGraph() : ComputationGraph<T>(), funcNodes() {}

    template <class T>
    T ScalerComputationGraph<T>::Forward()
    {
        if (funcNodes.IsEmpty())
            return 0;
        return funcNodes[funcNodes.Size() - 1]->Forward();
    }

    template <class T>
    void ScalerComputationGraph<T>::Backward()
    {
        if (funcNodes.IsEmpty())
            return;
        const std::size_t nNodes = funcNodes.Size();
        funcNodes[nNodes - 1]->UpdateGradient(1);

        std::size_t i = nNodes - 1;
        while (true)
        {
            auto input1 = funcNodes[i]->GetFirstInput();
            auto input2 = funcNodes[i]->GetSecondInput();
            auto gradients = funcNodes[i]->Backward();
            input1->UpdateGradient(gradients[0] * funcNodes[i]->GetPartialGradient());
            input2->UpdateGradient(gradients[1] * funcNodes[i]->GetPartialGradient());
            if (i == 0)
                break;
            i--;
        }
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->nodes.Size(); i++)
            this->nodes[i]->MarkGradientValuated();
    }

    template <class T>
    void ScalerComputationGraph<T>::SetValue(const ComputationGraphNodeHandler<T> &handler, const T &newValue) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::InvalidArgument(
                "ComputationGraph: Node could not be found.");
        try
        {
            auto variable = dynamic_cast<VariableNode *>(this->nodes[handler.index]);
            variable->SetValue(newValue);
            this->reset();
        }
        catch (const std::bad_cast &)
        {
            throw Exceptions::InvalidArgument(
                "ComputationGraph: The handler does not represent a variable node.");
        }
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateConstantNode(const T &value, const std::string &name)
    {
        ConstantNode *newNode = new ConstantNode(value, name);
        auto nodeIndex = this->nodes.Size();
        this->nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateVariableNode(const T &value, const std::string &name)
    {
        VariableNode *newNode = new VariableNode(value, name);
        auto nodeIndex = this->nodes.Size();
        this->nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex, true);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateFunctionNode(
        const ComputationGraphNodeHandler<T> &inputNodeHandler1,
        const ComputationGraphNodeHandler<T> &inputNodeHandler2,
        const GraphOperation &operation,
        const std::string &name)
    {
        FunctionNode *newNode;
        std::size_t input1Index = inputNodeHandler1.index;
        std::size_t input2Index = inputNodeHandler2.index;
        std::size_t newNodeIndex = this->nodes.Size();
        switch (operation)
        {
        case GraphOperation::Addition:
            newNode = new AddNode(this->nodes[input1Index], this->nodes[input2Index], name);
            break;
        case GraphOperation::Subtraction:
            newNode = new MinusNode(this->nodes[input1Index], this->nodes[input2Index], name);
            break;
        case GraphOperation::Multiplication:
            newNode = new MultiplyNode(this->nodes[input1Index], this->nodes[input2Index], name);
            break;
        case GraphOperation::Division:
            newNode = new DivideNode(this->nodes[input1Index], this->nodes[input2Index], name);
            break;
        case GraphOperation::Power:
            newNode = new PowerNode(this->nodes[input1Index], this->nodes[input2Index], name);
            break;
        default:
            throw Exceptions::InvalidArgument("ComputationGraph: Operation not recognized.");
        }
        this->nodes.Append(newNode);
        funcNodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, newNodeIndex);
    }
    template <class T>
    ScalerComputationGraph<T>::ScalerComputationGraphNode::ScalerComputationGraphNode(const std::string &nodeName) : ComputationGraph<T>::ComputationGraphNode(nodeName) {}

    template <class T>
    void ScalerComputationGraph<T>::ScalerComputationGraphNode::Reset()
    {
        this->gradient = 0;
        this->gradientValuated = false;
    }

    template <class T>
    void ScalerComputationGraph<T>::ScalerComputationGraphNode::UpdateGradient(const T &partialGradient)
    {
        this->gradient += partialGradient;
    }

    template <class T>
    ScalerComputationGraph<T>::ConstantNode::ConstantNode(const T &value, const std::string &nodeName) : ScalerComputationGraphNode(nodeName), value(value) {}

    template <class T>
    T ScalerComputationGraph<T>::ConstantNode::Forward() { return this->value; }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::ConstantNode::Backward() { return Tuple<T>({1}); }

    template <class T>
    ScalerComputationGraph<T>::VariableNode::VariableNode(const T &value, const std::string &nodeName) : ConstantNode(value, nodeName) {}

    template <class T>
    void ScalerComputationGraph<T>::VariableNode::SetValue(const T &newValue)
    {
        this->value = newValue;
    }

    template <class T>
    ScalerComputationGraph<T>::FunctionNode::FunctionNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName) : ScalerComputationGraphNode(nodeName), firstInput(input1), secondInput(input2) {}

    template <class T>
    typename ComputationGraph<T>::ComputationGraphNode *ScalerComputationGraph<T>::FunctionNode::GetFirstInput() const { return firstInput; }

    template <class T>
    typename ComputationGraph<T>::ComputationGraphNode *ScalerComputationGraph<T>::FunctionNode::GetSecondInput() const { return secondInput; }

    template <class T>
    ScalerComputationGraph<T>::AddNode::AddNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::AddNode::Forward()
    {
        return this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::AddNode::Backward() { return Tuple<T>({1, 1}); }

    template <class T>
    ScalerComputationGraph<T>::MinusNode::MinusNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::MinusNode::Forward()
    {
        return this->firstInput->Forward() - this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::MinusNode::Backward() { return Tuple<T>({1, -1}); }

    template <class T>
    ScalerComputationGraph<T>::MultiplyNode::MultiplyNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::MultiplyNode::Forward()
    {
        return this->firstInput->Forward() * this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::MultiplyNode::Backward()
    {
        return Tuple<T>({this->secondInput->Forward(), this->firstInput->Forward()});
    }

    template <class T>
    ScalerComputationGraph<T>::DivideNode::DivideNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::DivideNode::Forward()
    {
        return this->firstInput->Forward() / this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::DivideNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        return Tuple<T>({1 / secondInputValue,
                         -firstInputValue / (secondInputValue * secondInputValue)});
    }

    template <class T>
    ScalerComputationGraph<T>::PowerNode::PowerNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::PowerNode::Forward()
    {
        return Math::Power(this->firstInput->Forward(), this->secondInput->Forward());
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::PowerNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        const T powered = Math::Power(firstInputValue, secondInputValue - 1);
        return Tuple<T>({secondInputValue * powered,
                         powered * firstInputValue * Math::Log(firstInputValue)});
    }
}

#endif
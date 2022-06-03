#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cassert>

size_t sum(const size_t *values, size_t count)
{
  size_t sum = 0;
  while (count-- > 0)
    sum += values[count];

  return sum;
}

double sigmoid(double x)
{
  return 1.0 / (exp(-x) + 1);
}

double dsigmoid(double y)
{
  return y * (1 - y);
}

struct Mat
{
  double *data;
  size_t rows,
    cols;

  double &operator()(size_t row, size_t column);

  double operator()(size_t row, size_t column) const;
};

double &Mat::operator()(size_t row, size_t column)
{
  assert(row < rows && column < cols);

  return data[row * cols + column];
}

double Mat::operator()(size_t row, size_t column) const
{
  assert(row < rows && column < cols);

  return data[row * cols + column];
}

struct Vector
{
  double *data;
  size_t count;

  double &operator[](size_t index);

  double operator[](size_t index) const;
};

double &Vector::operator[](size_t index)
{
  assert(index < count);

  return data[index];
}

double Vector::operator[](size_t index) const
{
  assert(index < count);

  return data[index];
}

double cost(const Vector &actual, const Vector &expected)
{
  assert(actual.count == expected.count);

  double cost_value = 0;

  for (size_t i = 0; i < actual.count; i++)
  {
    double const diff = actual[i] - expected[i];
    cost_value += diff * diff;
  }

  return cost_value / actual.count;
}

void multiply_in_place(
  Vector &dest, const Mat &matrix, const Vector &vector
  )
{
  assert(matrix.cols == vector.count && dest.count == matrix.rows);

  for (size_t i = 0; i < matrix.rows; i++)
  {
    dest[i] = 0;
    for (size_t j = 0; j < matrix.cols; j++)
      dest[i] += matrix(i, j) * vector[j];
  }
}

void map(double (*func)(double), Vector &dest)
{
  for (size_t i = 0; i < dest.count; i++)
    dest[i] = func(dest[i]);
}

struct MemoryPool
{
  char *data;
  size_t size;
  size_t reserved;

  static MemoryPool allocate(size_t bytes);

  void *reserve(size_t bytes);
};

MemoryPool MemoryPool::allocate(size_t bytes)
{
  return { new char[bytes], 0, bytes };
}

void *MemoryPool::reserve(size_t bytes)
{
  if (size + bytes > reserved)
  {
    std::cerr << "ERROR: failed to reserve "
              << bytes
              << " bytes: out of reserved memory.\n";
    std::exit(EXIT_FAILURE);
  }

  void *const begin = (void *)(data + size);

  size += bytes;

  return begin;
}

struct Dataset
{
  double *data;
  size_t samples,
    inputs,
    outputs;

  static Dataset read(const char *filepath);

  Vector input(size_t row) const;

  Vector output(size_t row) const;
};

Vector Dataset::input(size_t row) const
{
  assert(row < samples);

  return { data + row * (inputs + outputs), inputs };
}

Vector Dataset::output(size_t row) const
{
  assert(row < samples);

  return { data + row * (inputs + outputs) + inputs, outputs };
}

Dataset Dataset::read(const char *filepath)
{
  std::fstream file(filepath, std::fstream::in);

  if (!file.is_open())
  {
    std::cerr << "ERROR: failed to open the file `"
              << filepath
              << "`.\n";
    std::exit(EXIT_FAILURE);
  }

  Dataset dataset;

  file >> dataset.samples
       >> dataset.inputs
       >> dataset.outputs;

  size_t const doubles_to_read =
    dataset.samples * (dataset.inputs + dataset.outputs);

  dataset.data = new double[doubles_to_read];

  size_t count = 0;
  while (!file.eof() && count < doubles_to_read)
  {
    file >> dataset.data[count];
    count++;
  }

  if (count < doubles_to_read)
  {
    std::cerr << "ERROR: read less values than indicated in the header of "
      "the file. Missing "
              << doubles_to_read - count
              << " values.\n";
    std::exit(EXIT_FAILURE);
  }

  file.close();

  return dataset;
}

struct NeuralNetwork
{
  Mat *weights;
  Vector *actvs;
  size_t *neuron_counts;
  size_t layers_count;
  size_t max_matrix_size;
  MemoryPool pool;

  static NeuralNetwork create(const size_t *neuron_counts, size_t layers);

  void feedforward(const Vector &input);

  void backpropagate(
    const Dataset &samples, double step, double eps
    );

  const Vector &input() const;

  const Vector &output() const;
};

double rand_range(double min, double max)
{
  return (double)std::rand() / RAND_MAX * (max - min) + min;
}

NeuralNetwork NeuralNetwork::create(const size_t *neuron_counts, size_t layers)
{
  if (layers <= 1)
  {
    std::cerr << "ERROR: number of layers should be more than 1.\n";
    std::exit(EXIT_FAILURE);
  }

  assert(neuron_counts[0] > 0);

  size_t max_network_size = neuron_counts[0] * sizeof (double);
  size_t max_delta_matrix_size = 0;

  for (size_t i = 1; i < layers; i++)
  {
    assert(neuron_counts[i] > 0);

    size_t const matrix_size =
      neuron_counts[i - 1] * neuron_counts[i] * sizeof (double);

    max_network_size += matrix_size + neuron_counts[i] * sizeof (double);
    max_delta_matrix_size = std::max(max_delta_matrix_size, matrix_size);
  }

  NeuralNetwork net;
  size_t const sizes[4] = {
    (layers - 1) * sizeof (Mat),
    layers * sizeof (Vector),
    layers * sizeof (size_t),
    max_network_size
  };

  size_t const total_size = sum(sizes, sizeof (sizes) / sizeof (*sizes));

  net.pool = MemoryPool::allocate(total_size);
  net.weights = (Mat *)net.pool.reserve(sizes[0]);
  net.actvs = (Vector *)net.pool.reserve(sizes[1]);
  net.neuron_counts = (size_t *)net.pool.reserve(sizes[2]);
  net.layers_count = layers;
  net.max_matrix_size = max_delta_matrix_size;

  for (double *i = (double *)(net.pool.data + net.pool.size),
         *const end = i + max_network_size / sizeof (double);
       i < end;
       i++)
  {
    *i = rand_range(-1, 1);
  }

  net.actvs[0].count = neuron_counts[0];
  net.actvs[0].data = (double *)net.pool.reserve(
    net.actvs[0].count * sizeof (double)
    );

  for (size_t i = 1; i < layers; i++)
  {
    auto &weights = net.weights[i - 1];

    weights.rows = neuron_counts[i];
    weights.cols = neuron_counts[i - 1];
    weights.data = (double *)net.pool.reserve(
      weights.rows * weights.cols * sizeof (double)
      );

    auto &actv = net.actvs[i];

    actv.count = neuron_counts[i];
    actv.data = (double *)net.pool.reserve(
      actv.count * sizeof (double)
      );
  }

  std::memcpy(net.neuron_counts, neuron_counts, layers * sizeof (size_t));

  assert(net.pool.size == net.pool.reserved);

  return net;
}

void NeuralNetwork::feedforward(const Vector &input)
{
  assert(neuron_counts[0] == input.count);

  std::memcpy(actvs[0].data, input.data, input.count * sizeof (double));

  for (size_t i = 1; i < layers_count; i++)
  {
    // Maybe should apply all of those operations at once.
    multiply_in_place(actvs[i], weights[i - 1], actvs[i - 1]);
    map(sigmoid, actvs[i]);
  }
}

const Vector &NeuralNetwork::output() const
{
  return actvs[layers_count - 1];
}

void NeuralNetwork::backpropagate(
  const Dataset &dataset, double step, double eps
  )
{
  char *const data = new char[2 * max_matrix_size];

  Mat front = { (double *)data, 0, 0 };
  Mat back = { (double *)(data + max_matrix_size), 0, 0 };

  auto const adjust_weights =
    [this, &front, &back, step](const Vector &expected)
  {
    {
      const Vector &actual = this->output();
      size_t const last_layer = layers_count - 1;

      front.rows = neuron_counts[last_layer];
      front.cols = neuron_counts[last_layer - 1];

      for (size_t row = 0; row < weights[last_layer - 1].rows; row++)
      {
        double const delta =
          (actual[row] - expected[row]) * dsigmoid(actual[row]);

        for (size_t col = 0; col < weights[last_layer - 1].cols; col++)
        {
          double &weight = weights[last_layer - 1](row, col);

          front(row, col) = weight * delta;
          weight -= step * delta * actvs[last_layer - 1][col];
        }
      }
    }

    std::swap(front, back);

    for (size_t layer = layers_count - 1; layer-- > 1; )
    {
      front.rows = neuron_counts[layer];
      front.cols = neuron_counts[layer - 1];

      for (size_t row = 0; row < weights[layer - 1].rows; row++)
      {
        double delta = 0;
        for (size_t k = 0; k < neuron_counts[layer + 1]; k++)
          delta += back(k, row);

        delta *= dsigmoid(actvs[layer][row]);

        for (size_t col = 0; col < weights[layer - 1].cols; col++)
        {
          double &weight = weights[layer - 1](row, col);

          front(row, col) = weight * delta;
          weight -= step * delta * actvs[layer - 1][col];
        }
      }

      std::swap(front, back);
    }
  };

  double error;
  do
  {
    error = 0;
    for (size_t i = 0; i < dataset.samples; i++)
    {
      feedforward(dataset.input(i));
      error += cost(this->output(), dataset.output(i));
      adjust_weights(dataset.output(i));
    }

    error /= dataset.samples;
  } while (error > eps);

  delete[] data;
}

int main(int argc, char **argv)
{
  assert(argc >= 3);

  std::srand(10234);

  Dataset dataset = Dataset::read(argv[1]);

  size_t const counts[4] = { dataset.inputs, 4, 3, dataset.outputs };
  NeuralNetwork net = NeuralNetwork::create(counts, 4);

  net.backpropagate(dataset, 0.05, 0.00001);

  delete[] dataset.data;

  dataset = Dataset::read(argv[2]);

  {
    double average_cost = 0;

    for (size_t i = 0; i < dataset.samples; i++)
    {
      net.feedforward(dataset.input(i));
      average_cost += cost(net.output(), dataset.output(i));
    }

    std::cout << "average cost: " << average_cost / dataset.samples << '\n';
  }

  delete[] dataset.data;
  delete[] net.pool.data;
}

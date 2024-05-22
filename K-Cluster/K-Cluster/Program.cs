using System;
using System.Collections.Generic;
using System.IO;

public class Program
{
    public static void Main(string[] args)
    {
        Console.Write("Enter the number of clusters (k): ");
        int k = int.Parse(Console.ReadLine());
        var data = LoadData("C:\\Users\\piotr\\Desktop\\GitHub\\NAI\\NAI-MMP5\\K-Cluster\\K-Cluster\\iris.data");

        var (assignments, centroids) = KMeans(data, k);

        PrintClusters(data, assignments, k);
    }

    private static List<double[]> LoadData(string filePath)
    {
        var data = new List<double[]>();
        using (var reader = new StreamReader(filePath))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                var tokens = line.Split(',');
                var features = new double[tokens.Length - 1];
                for (int i = 0; i < tokens.Length - 1; i++)
                {
                    features[i] = double.Parse(tokens[i]);
                }
                data.Add(features);
            }
        }
        return data;
    }

    private static (int[], double[][]) KMeans(List<double[]> data, int k, int maxIterations = 100)
    {
        Random rnd = new Random();
        double[][] centroids = ChooseInitialCentroids(data, k, rnd);
        int[] assignments = new int[data.Count];
        bool changed;

        do
        {
            changed = false;

            // Assign clusters
            for (int i = 0; i < data.Count; i++)
            {
                int bestCluster = FindNearestCentroid(data[i], centroids);
                if (assignments[i] != bestCluster)
                {
                    assignments[i] = bestCluster;
                    changed = true;
                }
            }

            // Recalculate centroids
            centroids = RecalculateCentroids(data, assignments, k);

            Console.WriteLine($"Total Distance: {TotalDistance(data, assignments, centroids):F2}");
        } while (changed && --maxIterations > 0);

        return (assignments, centroids);
    }

    private static double[][] ChooseInitialCentroids(List<double[]> data, int k, Random rnd)
    {
        double[][] centroids = new double[k][];
        var indices = new HashSet<int>();
        for (int i = 0; i < k; i++)
        {
            int index;
            do
            {
                index = rnd.Next(data.Count);
            } while (!indices.Add(index));
            centroids[i] = (double[])data[index].Clone();
        }
        return centroids;
    }

    private static int FindNearestCentroid(double[] dataPoint, double[][] centroids)
    {
        double minDistance = double.MaxValue;
        int clusterIndex = -1;
        for (int i = 0; i < centroids.Length; i++)
        {
            double distance = EuclideanDistance(dataPoint, centroids[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                clusterIndex = i;
            }
        }
        return clusterIndex;
    }

    private static double EuclideanDistance(double[] point1, double[] point2)
    {
        double sum = 0;
        for (int i = 0; i < point1.Length; i++)
        {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double[][] RecalculateCentroids(List<double[]> data, int[] assignments, int k)
    {
        double[][] centroids = new double[k][];
        int[] counts = new int[k];
        for (int i = 0; i < k; i++)
        {
            centroids[i] = new double[data[0].Length];
        }

        for (int i = 0; i < assignments.Length; i++)
        {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int j = 0; j < data[i].Length; j++)
            {
                centroids[cluster][j] += data[i][j];
            }
        }

        for (int i = 0; i < k; i++)
        {
            if (counts[i] == 0) continue;
            for (int j = 0; j < centroids[i].Length; j++)
            {
                centroids[i][j] /= counts[i];
            }
        }

        return centroids;
    }

    private static double TotalDistance(List<double[]> data, int[] assignments, double[][] centroids)
    {
        double total = 0;
        for (int i = 0; i < data.Count; i++)
        {
            total += EuclideanDistance(data[i], centroids[assignments[i]]);
        }
        return total;
    }

    private static void PrintClusters(List<double[]> data, int[] assignments, int k)
    {
        for (int i = 0; i < k; i++)
        {
            Console.WriteLine($"\nCluster {i + 1}:");
            foreach (var point in data)
            {
                if (assignments[data.IndexOf(point)] == i)
                {
                    Console.WriteLine(String.Join(", ", point));
                }
            }
        }
    }
}

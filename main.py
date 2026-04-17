from src.penguins_analysis import AnalysisConfig, run_full_analysis


def main() -> None:
    result = run_full_analysis(AnalysisConfig())

    print("Analise concluida com sucesso.")
    print(f"Registros: {result['df'].height}")
    print("\nContagem por ilha:")
    print(result["island_counts"])
    print("\nContagem por especie:")
    print(result["species_counts"])
    print("\nResumo de faltantes:")
    print(result["missing_summary"])
    print(f"\nParquet gerado: {result['clean_parquet']}")
    print("Graficos gerados:")
    for p in result["graphs"]:
        print(f"- {p}")


if __name__ == "__main__":
    main()

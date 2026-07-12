def test_dashboard_module_imports_without_error():
    """
    Dashboard modülünün en azından import edilebildiğini doğrular
    (syntax hatası, eksik import gibi temel sorunları yakalar).
    Not: st.set_page_config() Streamlit context'i dışında çağrılırsa
    hata verebilir, bu yüzden bu testi dashboard'u refactor ettikten
    sonra (fonksiyonlara böldükten sonra) genişletmek daha sağlıklı olur.
    """
    import importlib

    module = importlib.import_module("algorithm_arena.app.registry")
    assert hasattr(module, "OPTIMIZER_REGISTRY")
    assert hasattr(module, "BENCHMARK_REGISTRY")
    assert len(module.OPTIMIZER_REGISTRY) == 6

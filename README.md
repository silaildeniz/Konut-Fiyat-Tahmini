# Konut-Fiyat-Tahmini
LSTM Kullanarak Özelliklere Göre Konut Fiyatı Tahmin Eden Uygulama
# Konut Fiyat Tahmin Projesi

Bu proje, İstanbul'daki ilçeler için gelecekteki konut fiyatlarını tahmin etmeyi amaçlayan bir makine öğrenimi uygulamasıdır. Proje, PyCharm kullanılarak geliştirilmiş ve veri işleme, model eğitimi ve tahmin süreçlerini içermektedir.

## Özellikler

- **Kullanıcı Seçimi:** Kullanıcılar belirli bir şehri seçebilir.
- **Veri İşleme:** Veriler temizlenir ve analize uygun hale getirilir. "Room" sütunundaki "2 + 1" formatındaki değerler, oda ve salon sayısı olarak ayrıştırılır.
- **LSTM Modeli:** Uzun vadeli kısa dönemli bellek (LSTM) ağları kullanılarak gelecekteki konut fiyat değişim oranları tahmin edilir.
- **Sonuçlar:** Seçilen şehir ve girilen özelliklere göre konut fiyatını tahmin eder.

## Kullanılan Teknolojiler

- Python 3.10
- Pandas
- NumPy
- TensorFlow (LSTM modeli için)
- Matplotlib ve Seaborn (veri görselleştirme için)



## Veri Seti

Veri seti, İstanbul'daki konut fiyatlarını, oda ve salon sayılarını, konutun bulunduğu şehri ve diğer özellikleri içerir. Örnek veri:

| Price      | Room  | Area | Age | Location | Floor |
|------------|-------|------|-----|----------|-------|
| 5200000    | 2 + 1 | 250  | 50  | Adalar   | 2     |
| 1700000    | 2 + 1 | 115  | 30  | Adalar   | 2     |
| 23500000   | 3 + 1 | 430  | 0   | Adalar   | 4     |
Veri Seti Kaggle'dan alınmıştır.


app.py dosyası çalışmaktadır.



## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

Eğer herhangi bir sorunuz varsa, lütfen bana ulaşın: [silaildeniz10@gmail.com].


